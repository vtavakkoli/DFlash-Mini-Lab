from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class V11Config:
    """Training-free Boltzmann uncertainty gate for sparse MOBS correction."""

    temperature: float = 0.08
    gate_floor: float = 0.12
    mobs_budget: int = 2
    top_k: int = 4
    pair_weight: float = 1.0
    early_position_bias: float = 0.15


def _alt_probability(margin: float, temperature: float) -> float:
    temp = max(1e-5, float(temperature))
    z = min(60.0, max(0.0, float(margin)) / temp)
    return 1.0 / (1.0 + math.exp(z))


def select_v11_boltzmann_gated_mobs(runtime, draft_logits: np.ndarray, context: np.ndarray, prev_token: int, config: V11Config):
    """Route only uncertain DFlash slots through sparse MOBS correction.

    Boltzmann is deterministic here: the two-way alternative probability is an
    uncertainty signal, not a sampling coin. Confident slots remain DFlash
    argmax. Exact target verification remains authoritative downstream.
    """
    _, top_ids, top_vals = runtime._top_k(draft_logits, int(config.top_k))
    block = int(top_ids.shape[0])
    if block == 0:
        return np.empty(0, dtype=np.int64), {"pair_scores": 0, "gated_positions": 0, "eligible_positions": 0, "fast_argmax_positions": 0, "mean_uncertainty": 0.0}

    chosen = top_ids[:, 0].astype(np.int64, copy=True)
    if top_ids.shape[1] < 2 or int(config.mobs_budget) <= 0:
        return chosen, {"pair_scores": 0, "gated_positions": 0, "eligible_positions": 0, "fast_argmax_positions": block, "mean_uncertainty": 0.0}

    margins = (top_vals[:, 0] - top_vals[:, 1]).astype(np.float64)
    uncertainty = np.asarray([_alt_probability(float(m), float(config.temperature)) for m in margins], dtype=np.float64)
    priority = uncertainty / (1.0 + float(config.early_position_bias) * np.arange(block, dtype=np.float64))
    eligible = [i for i in range(block) if float(uncertainty[i]) >= float(config.gate_floor)]
    eligible.sort(key=lambda i: (-float(priority[i]), i))
    gated = eligible[: max(0, min(int(config.mobs_budget), len(eligible)))]

    if not gated:
        return chosen, {
            "pair_scores": 0,
            "gated_positions": 0,
            "eligible_positions": int(len(eligible)),
            "fast_argmax_positions": block,
            "mean_uncertainty": float(np.mean(uncertainty)),
            "max_uncertainty": float(np.max(uncertainty)),
        }

    gate, a, b = runtime._selector_state(context)
    scale = float(runtime.selector_scale)
    pair_scores = 0

    def forward(previous: int, candidate_ids: np.ndarray) -> np.ndarray:
        nonlocal pair_scores
        ids = np.asarray(candidate_ids, dtype=np.int64)
        pair_scores += int(ids.size)
        pa = a[int(previous)] * gate
        return (b[ids] * pa[None, :]).sum(axis=-1) * scale

    def backward(candidate_ids: np.ndarray, following: int) -> np.ndarray:
        nonlocal pair_scores
        ids = np.asarray(candidate_ids, dtype=np.int64)
        pair_scores += int(ids.size)
        cb = b[int(following)]
        return ((a[ids] * gate[None, :]) * cb[None, :]).sum(axis=-1) * scale

    center = (block - 1) / 2.0
    visit = sorted(gated, key=lambda p: (abs(float(p) - center), p))
    for pos in visit:
        candidates = top_ids[pos]
        scores = top_vals[pos].astype(np.float64, copy=True)
        previous = int(prev_token) if pos == 0 else int(chosen[pos - 1])
        scores += float(config.pair_weight) * forward(previous, candidates)
        if pos + 1 < block:
            scores += float(config.pair_weight) * backward(candidates, int(chosen[pos + 1]))
        chosen[pos] = int(candidates[int(np.argmax(scores))])

    return chosen, {
        "pair_scores": int(pair_scores),
        "gated_positions": int(len(gated)),
        "eligible_positions": int(len(eligible)),
        "fast_argmax_positions": int(block - len(gated)),
        "mean_uncertainty": float(np.mean(uncertainty)),
        "max_uncertainty": float(np.max(uncertainty)),
    }


def v11_grid() -> tuple[V11Config, ...]:
    configs: list[V11Config] = []
    for temperature in (0.04, 0.08):
        for gate_floor in (0.08, 0.16):
            for mobs_budget in (1, 2, 3):
                for top_k in (4, 8):
                    configs.append(V11Config(temperature=temperature, gate_floor=gate_floor, mobs_budget=mobs_budget, top_k=top_k, pair_weight=1.0, early_position_bias=0.15))
    return tuple(configs)


def config_key(config: V11Config) -> str:
    return f"T={config.temperature:g}|gate={config.gate_floor:g}|budget={config.mobs_budget}|k={config.top_k}|pair={config.pair_weight:g}|early={config.early_position_bias:g}"


def config_dict(config: V11Config) -> dict:
    return asdict(config)
