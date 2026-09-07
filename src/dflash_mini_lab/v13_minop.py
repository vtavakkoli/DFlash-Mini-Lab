from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
import torch

from .lfm_benchmark import RealDecodeStats
from .lfm_v10 import V10Config


V13_METHOD = "minop_v13"
_MASK64 = (1 << 64) - 1


def _mix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (x ^ (x >> 31)) & _MASK64


def _uniform01(context: np.ndarray, prev_token: int, pos: int, top1: int, top2: int) -> float:
    ctx = np.asarray(context[: min(8, int(context.size))], dtype=np.float32)
    fingerprint = int(float(np.abs(ctx).sum()) * 1_000_000.0) & _MASK64
    x = fingerprint ^ ((int(prev_token) * 0x9E3779B1) & _MASK64)
    x ^= ((int(pos) + 1) * 0xD1B54A32D192ED03) & _MASK64
    x ^= ((int(top1) << 17) ^ int(top2)) & _MASK64
    x = _mix64(x)
    return (((x >> 11) & ((1 << 53) - 1)) + 0.5) / float(1 << 53)


@torch.inference_mode()
def draft_top2(runtime, context: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run the existing drafter but transfer only top-2 IDs/scores to NumPy.

    V10 computes full draft logits in NumPy and then applies a NumPy top-k.
    V13 performs top-2 selection while the logits are still Torch tensors, so
    only Bx2 values cross the Torch->NumPy boundary.
    """
    x = torch.from_numpy(np.asarray(context, dtype=np.float32)).unsqueeze(0)
    hidden = runtime.drafter.encode(x)
    logits = runtime.drafter.head(hidden)[0]
    k = min(2, int(logits.shape[-1]))
    vals, local = torch.topk(logits, k=k, dim=-1, largest=True, sorted=True)
    ids = runtime.candidate_ids_t[local.cpu()]
    return (
        ids.numpy().astype(np.int64, copy=False),
        vals.float().cpu().numpy().astype(np.float32, copy=False),
    )


def select_v13_minop(
    top_ids: np.ndarray,
    top_vals: np.ndarray,
    context: np.ndarray,
    prev_token: int,
    config: V10Config,
) -> tuple[np.ndarray, dict]:
    """V10 decision policy with a hard one-slot routing budget.

    The common path is exactly DFlash top-1. Only the single least-confident
    eligible slot can take the top-2 branch.
    """
    ids = np.asarray(top_ids, dtype=np.int64)
    vals = np.asarray(top_vals, dtype=np.float32)
    block = int(ids.shape[0])
    if block == 0:
        return np.empty(0, dtype=np.int64), {
            "candidate_scores": 0,
            "routed_positions": 0,
            "fast_argmax_positions": 0,
            "top2_values_transferred": 0,
        }

    chosen = ids[:, 0].copy()
    if int(ids.shape[1]) < 2:
        return chosen, {
            "candidate_scores": 0,
            "routed_positions": 0,
            "fast_argmax_positions": block,
            "top2_values_transferred": int(ids.size),
        }

    margins = (vals[:, 0] - vals[:, 1]).astype(np.float64)
    eligible = np.flatnonzero(margins < float(config.margin_cutoff))
    if eligible.size == 0:
        return chosen, {
            "candidate_scores": 0,
            "routed_positions": 0,
            "fast_argmax_positions": block,
            "top2_values_transferred": int(ids.size),
            "mean_margin": float(np.mean(margins)),
        }

    weighted = margins[eligible] * (1.0 + 0.20 * eligible.astype(np.float64))
    pos = int(eligible[int(np.argmin(weighted))])
    margin = max(0.0, float(margins[pos]))
    temp = max(1e-5, float(config.temperature) / (1.0 + float(config.margin_slope) * margin))
    z = min(60.0, margin / temp)
    p_second = 1.0 / (1.0 + math.exp(z))
    t1, t2 = int(ids[pos, 0]), int(ids[pos, 1])
    if _uniform01(context, prev_token, pos, t1, t2) < p_second:
        chosen[pos] = t2

    return chosen, {
        "candidate_scores": 2,
        "routed_positions": 1,
        "fast_argmax_positions": block - 1,
        "top2_values_transferred": int(ids.size),
        "mean_margin": float(np.mean(margins)),
        "selected_margin": float(margins[pos]),
        "second_probability": float(p_second),
    }


def _verify(runtime, seq: np.ndarray, proposal: np.ndarray):
    verify_input = np.concatenate([seq, proposal])
    t0 = time.perf_counter()
    logits = runtime.target_logits(verify_input)
    elapsed = time.perf_counter() - t0
    prefix = int(seq.size)
    width = int(proposal.size)
    verifier = np.argmax(logits[prefix - 1 : prefix - 1 + width], axis=-1).astype(np.int64)
    mismatch = np.flatnonzero(proposal != verifier)
    accepted = width if mismatch.size == 0 else int(mismatch[0])
    return verifier, accepted, elapsed


def v13_decode(runtime, input_ids: np.ndarray, max_new_tokens: int, *, config: V10Config):
    seq = np.asarray(input_ids, dtype=np.int64).copy()
    start_len = int(seq.size)
    target_calls = draft_calls = accepted_total = proposed_total = 0
    target_seconds = context_seconds = draft_seconds = selection_seconds = 0.0
    candidate_scores = routed_positions = fast_positions = top2_transferred = 0
    wall0 = time.perf_counter()

    while int(seq.size) - start_len < int(max_new_tokens):
        remaining = int(max_new_tokens) - (int(seq.size) - start_len)

        t0 = time.perf_counter()
        context = runtime.context_features(seq)
        context_seconds += time.perf_counter() - t0

        t0 = time.perf_counter()
        top_ids, top_vals = draft_top2(runtime, context)
        draft_seconds += time.perf_counter() - t0
        draft_calls += 1

        t0 = time.perf_counter()
        full, meta = select_v13_minop(top_ids, top_vals, context, int(seq[-1]), config)
        proposal = np.asarray(full[: min(int(full.size), remaining)], dtype=np.int64)
        selection_seconds += time.perf_counter() - t0

        candidate_scores += int(meta.get("candidate_scores", 0))
        routed_positions += int(meta.get("routed_positions", 0))
        fast_positions += int(meta.get("fast_argmax_positions", 0))
        top2_transferred += int(meta.get("top2_values_transferred", 0))
        proposed_total += int(proposal.size)

        verifier, accepted, elapsed = _verify(runtime, seq, proposal)
        target_seconds += elapsed
        target_calls += 1
        accepted_total += int(accepted)

        if accepted:
            seq = np.concatenate([seq, proposal[:accepted]])
        if accepted < int(proposal.size) and int(seq.size) - start_len < int(max_new_tokens):
            seq = np.append(seq, verifier[accepted])

    seq = seq[: start_len + int(max_new_tokens)]
    wall = time.perf_counter() - wall0
    stats = RealDecodeStats(
        method=V13_METHOD,
        new_tokens=int(max_new_tokens),
        target_forward_passes=target_calls,
        draft_forward_passes=draft_calls,
        accepted_draft_tokens=accepted_total,
        proposed_draft_tokens=proposed_total,
        wall_seconds=wall,
        target_seconds=target_seconds,
        context_seconds=context_seconds,
        draft_seconds=draft_seconds,
        selection_seconds=selection_seconds,
        boltzmann_candidate_scores=int(candidate_scores),
    )
    meta = {
        "v13_candidate_scores": int(candidate_scores),
        "v13_routed_positions": int(routed_positions),
        "v13_fast_argmax_positions": int(fast_positions),
        "v13_top2_values_transferred": int(top2_transferred),
        "v13_policy_source": "selected V10 temperature/cutoff/slope; hard routing budget=1",
    }
    return seq, stats, meta
