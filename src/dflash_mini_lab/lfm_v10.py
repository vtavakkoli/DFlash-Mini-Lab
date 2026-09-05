from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


_MASK64 = (1 << 64) - 1


@dataclass(frozen=True)
class V10Config:
    """Runtime-only Advanced Boltzmann QuickPath configuration."""

    temperature: float = 0.08
    margin_cutoff: float = 0.20
    stochastic_budget: int = 1
    margin_slope: float = 1.5


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


def select_v10_quickpath(runtime, draft_logits: np.ndarray, context: np.ndarray, prev_token: int, config: V10Config):
    """Low-operation top-2 Boltzmann selection.

    Confident slots use DFlash argmax. Only a small uncertainty budget is
    sampled. Exact two-way Boltzmann sampling needs one logistic probability
    and one stateless uniform draw rather than top-k Gumbel scoring.
    """
    _, top_ids, top_vals = runtime._top_k(draft_logits, 2)
    block = int(top_ids.shape[0])
    if block == 0:
        return np.empty(0, dtype=np.int64), {"candidate_scores": 0, "uncertain_positions": 0, "sampled_positions": 0, "fast_argmax_positions": 0}

    chosen = top_ids[:, 0].astype(np.int64, copy=True)
    if top_ids.shape[1] < 2:
        return chosen, {"candidate_scores": 0, "uncertain_positions": 0, "sampled_positions": 0, "fast_argmax_positions": block}

    margins = (top_vals[:, 0] - top_vals[:, 1]).astype(np.float64)
    eligible = [i for i in range(block) if float(margins[i]) < float(config.margin_cutoff)]
    eligible.sort(key=lambda i: float(margins[i]) * (1.0 + 0.20 * i))
    budget = max(0, min(int(config.stochastic_budget), len(eligible)))
    sampled = eligible[:budget]

    candidate_scores = 0
    for pos in sampled:
        margin = max(0.0, float(margins[pos]))
        temp = max(1e-5, float(config.temperature) / (1.0 + float(config.margin_slope) * margin))
        z = min(60.0, margin / temp)
        p_second = 1.0 / (1.0 + math.exp(z))
        t1, t2 = int(top_ids[pos, 0]), int(top_ids[pos, 1])
        if _uniform01(context, prev_token, pos, t1, t2) < p_second:
            chosen[pos] = t2
        candidate_scores += 2

    meta = {
        "candidate_scores": int(candidate_scores),
        "uncertain_positions": int(len(eligible)),
        "sampled_positions": int(len(sampled)),
        "fast_argmax_positions": int(block - len(sampled)),
        "mean_margin": float(np.mean(margins)) if margins.size else 0.0,
    }
    return chosen, meta


def v10_grid() -> tuple[V10Config, ...]:
    configs = []
    for temperature in (0.03, 0.06, 0.10, 0.16):
        for cutoff in (0.05, 0.15, 0.35):
            for budget in (1, 2):
                configs.append(V10Config(temperature=temperature, margin_cutoff=cutoff, stochastic_budget=budget, margin_slope=1.5))
    return tuple(configs)


def successive_halving_plan(total_configs: int, calibration_prompts: int, survivors: int = 4, stage1_prompts: int = 2) -> dict:
    stage1 = min(max(1, int(stage1_prompts)), max(1, int(calibration_prompts)))
    keep = min(max(1, int(survivors)), max(1, int(total_configs)))
    full = int(total_configs) * int(calibration_prompts)
    used = int(total_configs) * stage1 + keep * int(calibration_prompts)
    return {
        "total_configs": int(total_configs),
        "stage1_prompts": stage1,
        "survivors": keep,
        "full_grid_prompt_config_evaluations": full,
        "successive_halving_prompt_config_evaluations": used,
        "calibration_work_reduction_fraction": 1.0 - (used / max(full, 1)),
        "training_steps": 0,
    }


def config_key(config: V10Config) -> str:
    return f"T={config.temperature:g}|cut={config.margin_cutoff:g}|budget={config.stochastic_budget}|slope={config.margin_slope:g}"


def config_dict(config: V10Config) -> dict:
    return asdict(config)
