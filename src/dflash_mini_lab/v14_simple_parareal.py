from __future__ import annotations

from dataclasses import dataclass
import json
import time
from pathlib import Path

import numpy as np

from .lfm_verification import trim_proposal, verify_draft as _verify

from .lfm_benchmark import RealDecodeStats
from .v13_minop import draft_top2


V14_METHOD = "simple_parareal_v14"


@dataclass(frozen=True)
class V14Estimator:
    intercept: float
    coarse_gap_weight: float
    position_weight: float
    damping: float = 1.0
    switch_threshold: float = 0.0
    max_corrections: int = 1
    metadata: dict | None = None


def save_estimator(path: str | Path, estimator: V14Estimator) -> None:
    payload = {
        "format_version": 1,
        "algorithm": "DFlash14-Simple-PARAREAL",
        "coefficients": {
            "intercept": float(estimator.intercept),
            "coarse_gap_weight": float(estimator.coarse_gap_weight),
            "position_weight": float(estimator.position_weight),
        },
        "config": {
            "damping": float(estimator.damping),
            "switch_threshold": float(estimator.switch_threshold),
            "max_corrections": int(estimator.max_corrections),
        },
        "metadata": estimator.metadata or {},
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_estimator(path: str | Path) -> V14Estimator:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    c = payload["coefficients"]
    cfg = payload.get("config", {})
    return V14Estimator(
        intercept=float(c["intercept"]),
        coarse_gap_weight=float(c["coarse_gap_weight"]),
        position_weight=float(c["position_weight"]),
        damping=float(cfg.get("damping", 1.0)),
        switch_threshold=float(cfg.get("switch_threshold", 0.0)),
        max_corrections=int(cfg.get("max_corrections", 1)),
        metadata=dict(payload.get("metadata", {})),
    )


def estimate_fine_gap(estimator: V14Estimator, coarse_gap: np.ndarray) -> np.ndarray:
    gap = np.asarray(coarse_gap, dtype=np.float64)
    block = int(gap.size)
    if block <= 1:
        position = np.zeros(block, dtype=np.float64)
    else:
        position = np.arange(block, dtype=np.float64) / float(block - 1)
    return (
        float(estimator.intercept)
        + float(estimator.coarse_gap_weight) * gap
        + float(estimator.position_weight) * position
    )


def select_v14_simple_parareal(
    top_ids: np.ndarray,
    top_vals: np.ndarray,
    estimator: V14Estimator,
) -> tuple[np.ndarray, dict]:
    """One-round scalar Parareal correction over top-2 score gaps.

    G is the DFlash coarse top1-top2 gap. The learned F-hat is a three-term
    affine estimate of the target's signed gap on those same two candidates.
    One Parareal-style update is applied:
        gap_1 = G + damping * (F_hat - G)
    Only the most strongly negative corrected slot may flip to top-2.
    """
    ids = np.asarray(top_ids, dtype=np.int64)
    vals = np.asarray(top_vals, dtype=np.float32)
    block = int(ids.shape[0])
    if block == 0:
        return np.empty(0, dtype=np.int64), {
            "estimator_ops": 0,
            "switched_positions": 0,
            "fast_argmax_positions": 0,
        }

    chosen = ids[:, 0].copy()
    if int(ids.shape[1]) < 2:
        return chosen, {
            "estimator_ops": 0,
            "switched_positions": 0,
            "fast_argmax_positions": block,
        }

    coarse_gap = (vals[:, 0] - vals[:, 1]).astype(np.float64)
    fine_hat = estimate_fine_gap(estimator, coarse_gap)
    corrected_gap = coarse_gap + float(estimator.damping) * (fine_hat - coarse_gap)

    eligible = np.flatnonzero(corrected_gap < float(estimator.switch_threshold))
    budget = min(max(0, int(estimator.max_corrections)), int(eligible.size))
    switched = 0
    if budget:
        order = eligible[np.argsort(corrected_gap[eligible])]
        for pos in order[:budget]:
            chosen[int(pos)] = int(ids[int(pos), 1])
            switched += 1

    estimator_ops = int(block * 6)
    return chosen, {
        "estimator_ops": estimator_ops,
        "switched_positions": int(switched),
        "fast_argmax_positions": int(block - switched),
        "mean_coarse_gap": float(np.mean(coarse_gap)),
        "mean_estimated_fine_gap": float(np.mean(fine_hat)),
        "mean_corrected_gap": float(np.mean(corrected_gap)),
        "mean_abs_residual": float(np.mean(np.abs(fine_hat - coarse_gap))),
    }



def v14_decode(runtime, input_ids: np.ndarray, max_new_tokens: int, *, estimator: V14Estimator):
    seq = np.asarray(input_ids, dtype=np.int64).copy()
    start_len = int(seq.size)
    target_calls = draft_calls = accepted_total = proposed_total = 0
    target_seconds = context_seconds = draft_seconds = selection_seconds = 0.0
    estimator_ops = switched_positions = fast_positions = 0
    residuals: list[float] = []
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
        full, meta = select_v14_simple_parareal(top_ids, top_vals, estimator)
        proposal = trim_proposal(runtime, full, remaining)
        selection_seconds += time.perf_counter() - t0

        estimator_ops += int(meta.get("estimator_ops", 0))
        switched_positions += int(meta.get("switched_positions", 0))
        fast_positions += int(meta.get("fast_argmax_positions", 0))
        residuals.append(float(meta.get("mean_abs_residual", 0.0)))
        proposed_total += int(proposal.size)

        verifier, accepted, elapsed = _verify(runtime, seq, proposal)
        target_seconds += elapsed
        target_calls += 1
        accepted_total += int(accepted)

        if accepted:
            seq = np.concatenate([seq, proposal[:accepted]])
        if accepted < int(verifier.size) and int(seq.size) - start_len < int(max_new_tokens):
            seq = np.append(seq, verifier[accepted])

    seq = seq[: start_len + int(max_new_tokens)]
    wall = time.perf_counter() - wall0
    stats = RealDecodeStats(
        method=V14_METHOD,
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
    )
    meta = {
        "v14_estimator_ops": int(estimator_ops),
        "v14_switched_positions": int(switched_positions),
        "v14_fast_argmax_positions": int(fast_positions),
        "v14_mean_abs_residual": float(np.mean(residuals)) if residuals else 0.0,
        "v14_damping": float(estimator.damping),
        "v14_switch_threshold": float(estimator.switch_threshold),
    }
    return seq, stats, meta
