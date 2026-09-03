from __future__ import annotations

import numpy as np

from .runtime import CpuReferenceRuntime, _top_k_ids_values

_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
_C1 = np.uint64(0x9E3779B97F4A7C15)
_C2 = np.uint64(0xBF58476D1CE4E5B9)
_C3 = np.uint64(0x94D049BB133111EB)


def _context_seed(context: np.ndarray, prev_token: int, salt: int = 0) -> np.uint64:
    x = int(float(np.abs(np.asarray(context[: min(12, context.size)], dtype=np.float64)).sum()) * 1_000_000.0)
    return np.uint64((x ^ (int(prev_token) * 0x9E3779B1) ^ int(salt)) & 0xFFFFFFFFFFFFFFFF)


def _gumbel_for_ids(ids: np.ndarray, seed: np.uint64, position: int) -> np.ndarray:
    """Deterministic SplitMix64 -> U(0,1) -> Gumbel noise."""
    x = np.asarray(ids, dtype=np.uint64) + seed + np.uint64(position + 1) * _C1
    x = (x ^ (x >> np.uint64(30))) * _C2 & _MASK64
    x = (x ^ (x >> np.uint64(27))) * _C3 & _MASK64
    x = x ^ (x >> np.uint64(31))
    # Use the high 53 bits just like a double-precision RNG.
    u = ((x >> np.uint64(11)).astype(np.float64) + 0.5) * (1.0 / (1 << 53))
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    return -np.log(-np.log(u))


def _adaptive_temperature(top_vals: np.ndarray, base_temperature: float) -> np.ndarray:
    """Use more exploration only where top-1/top-2 are close."""
    base = max(float(base_temperature), 1e-6)
    if top_vals.shape[-1] < 2:
        return np.full(top_vals.shape[0], 1e-6, dtype=np.float64)
    margin = np.maximum(0.0, top_vals[:, 0].astype(np.float64) - top_vals[:, 1].astype(np.float64))
    # exp(-margin) is near one for uncertain positions and quickly approaches zero
    # for confident positions; the floor makes those positions nearly argmax.
    return np.maximum(base * np.exp(-margin), base * 0.02)


def dflash6_boltzmann_select_path(
    draft_logits: np.ndarray,
    context: np.ndarray,
    prev_token: int,
    top_k: int = 8,
    temperature: float = 0.15,
) -> tuple[np.ndarray, int]:
    """Independent deterministic Boltzmann/Gumbel proposals from existing logits.

    No learned selector or extra model forward pass is used. Sampling is restricted
    to the drafter's retained top-k set, and temperature falls automatically when
    the draft margin is large.
    """
    top_ids, top_vals = _top_k_ids_values(draft_logits, top_k)
    if top_ids.size == 0:
        return np.empty(0, dtype=np.int64), 0
    temps = _adaptive_temperature(top_vals, temperature)
    seed = _context_seed(context, prev_token, salt=0xD6B017)
    chosen = np.empty(top_ids.shape[0], dtype=np.int64)
    for pos in range(top_ids.shape[0]):
        c = top_ids[pos]
        g = _gumbel_for_ids(c, seed, pos)
        scores = top_vals[pos].astype(np.float64) / temps[pos] + g
        chosen[pos] = int(c[int(np.argmax(scores))])
    return chosen, int(top_ids.size)


def dflash6_bmobs_select_path(
    runtime: CpuReferenceRuntime,
    draft_logits: np.ndarray,
    context: np.ndarray,
    prev_token: int,
    top_k: int = 8,
    temperature: float = 0.15,
) -> tuple[np.ndarray, int, int]:
    """Boltzmann middle anchor followed by the existing O(BK) MOBS gap fill."""
    top_ids, top_vals = _top_k_ids_values(draft_logits, top_k)
    block = int(top_ids.shape[0])
    if block == 0:
        return np.empty(0, dtype=np.int64), 0, 0

    # For even blocks choose the more uncertain of the two middle positions.
    centers = [block // 2] if block % 2 else [block // 2 - 1, block // 2]
    if len(centers) == 2 and top_vals.shape[1] > 1:
        margins = [float(top_vals[p, 0] - top_vals[p, 1]) for p in centers]
        anchor = centers[int(np.argmin(margins))]
    else:
        anchor = centers[0]

    temps = _adaptive_temperature(top_vals, temperature)
    c = top_ids[anchor]
    seed = _context_seed(context, prev_token, salt=0xB00B5)
    scores = top_vals[anchor].astype(np.float64) / temps[anchor] + _gumbel_for_ids(c, seed, anchor)
    chosen = np.full(block, -1, dtype=np.int64)
    chosen[anchor] = int(c[int(np.argmax(scores))])
    chosen, pair_scores = runtime._anchored_gap_fill(top_ids, top_vals, chosen, context, prev_token)
    return chosen, int(pair_scores), int(c.size)
