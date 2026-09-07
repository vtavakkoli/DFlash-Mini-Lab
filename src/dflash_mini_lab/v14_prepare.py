from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from .lfm_prepare import _read_list
from .lfm_runtime import LfmReferenceRuntime
from .v12_prepare import collect_regression_examples
from .v14_simple_parareal import V14Estimator, save_estimator


def _matrix(examples) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    targets = []
    for ex in examples:
        coarse = np.asarray(ex.coarse_scores, dtype=np.float64)
        fine = np.asarray(ex.fine_scores, dtype=np.float64)
        if coarse.shape[1] < 2 or fine.shape[1] < 2:
            continue
        block = int(coarse.shape[0])
        pos = np.zeros(block, dtype=np.float64) if block <= 1 else np.arange(block, dtype=np.float64) / float(block - 1)
        coarse_gap = coarse[:, 0] - coarse[:, 1]
        fine_gap = fine[:, 0] - fine[:, 1]
        rows.append(np.column_stack([np.ones(block), coarse_gap, pos]))
        targets.append(fine_gap)
    if not rows:
        raise RuntimeError("no V14 top-2 gap examples were constructed")
    return np.concatenate(rows, axis=0), np.concatenate(targets, axis=0)


def _metrics(estimator: V14Estimator, examples) -> dict:
    X, y = _matrix(examples)
    fine_hat = X @ np.asarray(
        [estimator.intercept, estimator.coarse_gap_weight, estimator.position_weight],
        dtype=np.float64,
    )
    coarse_gap = X[:, 1]
    corrected = coarse_gap + float(estimator.damping) * (fine_hat - coarse_gap)
    truth_top1 = y >= 0.0
    predicted_top1 = corrected >= float(estimator.switch_threshold)
    baseline_top1 = coarse_gap >= 0.0
    negative = ~truth_top1
    predicted_flip = ~predicted_top1
    tp = int(np.sum(predicted_flip & negative))
    fp = int(np.sum(predicted_flip & ~negative))
    fn = int(np.sum(~predicted_flip & negative))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "rows": int(y.size),
        "fine_gap_mse": float(np.mean((fine_hat - y) ** 2)),
        "fine_gap_mae": float(np.mean(np.abs(fine_hat - y))),
        "corrected_sign_accuracy": float(np.mean(predicted_top1 == truth_top1)),
        "coarse_sign_accuracy": float(np.mean(baseline_top1 == truth_top1)),
        "teacher_prefers_top2_rate": float(np.mean(negative)),
        "predicted_switch_rate": float(np.mean(predicted_flip)),
        "switch_precision": float(precision),
        "switch_recall": float(recall),
        "mean_coarse_gap": float(np.mean(coarse_gap)),
        "mean_fine_gap": float(np.mean(y)),
        "mean_estimated_fine_gap": float(np.mean(fine_hat)),
    }


def prepare_v14(
    *,
    aux_path: str | Path,
    output_path: str | Path,
    seeds_path: str | Path,
    max_seed_count: int = 24,
    generation_tokens: int = 24,
    ridge: float = 1e-3,
    damping: float = 1.0,
    switch_threshold: float = 0.0,
    max_corrections: int = 1,
    holdout_fraction: float = 0.2,
    cpu_threads: int = 2,
    seed: int = 29,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, int(cpu_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    runtime = LfmReferenceRuntime(aux_path, cpu_threads=cpu_threads)
    prompts = _read_list(seeds_path, "seeds")[: max(2, int(max_seed_count))]
    examples = collect_regression_examples(
        runtime,
        prompts,
        generation_tokens=int(generation_tokens),
        top_k=2,
        stride=1,
    )
    split = int(round(len(examples) * (1.0 - float(holdout_fraction))))
    split = min(max(1, split), len(examples) - 1)
    train_examples = examples[:split]
    holdout_examples = examples[split:]

    X, y = _matrix(train_examples)
    reg = np.eye(X.shape[1], dtype=np.float64) * float(ridge)
    reg[0, 0] = 0.0
    lhs = X.T @ X + reg
    rhs = X.T @ y
    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(lhs) @ rhs

    metadata = {
        "algorithm": "DFlash14-Simple-PARAREAL",
        "mechanism": "one-round scalar Parareal correction using a three-term affine fine-gap estimator",
        "parareal_mapping": {
            "coarse_G": "DFlash top1-top2 draft gap",
            "fine_F_training_only": "target logit gap on the same two draft candidates",
            "fine_estimator": "F_hat = a + b*G + c*normalized_position",
            "inference_update": "gap_1 = G + damping*(F_hat-G)",
            "routing": "at most one corrected slot may switch from draft top1 to top2",
        },
        "model_id": runtime.model_id,
        "candidate_size": int(runtime.candidate_size),
        "block_size": int(runtime.block_size),
        "seed_count": len(prompts),
        "generation_tokens_per_seed": int(generation_tokens),
        "training_examples": len(train_examples),
        "holdout_examples": len(holdout_examples),
        "ridge": float(ridge),
        "target_weights_redistributed": False,
    }
    estimator = V14Estimator(
        intercept=float(beta[0]),
        coarse_gap_weight=float(beta[1]),
        position_weight=float(beta[2]),
        damping=float(damping),
        switch_threshold=float(switch_threshold),
        max_corrections=int(max_corrections),
        metadata=metadata,
    )
    train_metrics = _metrics(estimator, train_examples)
    holdout_metrics = _metrics(estimator, holdout_examples)
    metadata["train_metrics"] = train_metrics
    metadata["holdout_metrics"] = holdout_metrics
    estimator = V14Estimator(
        intercept=estimator.intercept,
        coarse_gap_weight=estimator.coarse_gap_weight,
        position_weight=estimator.position_weight,
        damping=estimator.damping,
        switch_threshold=estimator.switch_threshold,
        max_corrections=estimator.max_corrections,
        metadata=metadata,
    )
    save_estimator(output_path, estimator)

    result = {
        **metadata,
        "artifact": str(output_path),
        "coefficients": {
            "intercept": estimator.intercept,
            "coarse_gap_weight": estimator.coarse_gap_weight,
            "position_weight": estimator.position_weight,
        },
        "config": {
            "damping": estimator.damping,
            "switch_threshold": estimator.switch_threshold,
            "max_corrections": estimator.max_corrections,
        },
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Fit the DFlash14 simple fine-gap estimator on frozen LFM2.5 teacher trajectories")
    p.add_argument("--aux", default="lfm-artifacts/lfm_aux.pt")
    p.add_argument("--output", default="lfm-artifacts/v14_simple_parareal.json")
    p.add_argument("--seeds", default="real_benchmarks/train_seeds.json")
    p.add_argument("--max-seed-count", type=int, default=24)
    p.add_argument("--generation-tokens", type=int, default=24)
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--damping", type=float, default=1.0)
    p.add_argument("--switch-threshold", type=float, default=0.0)
    p.add_argument("--max-corrections", type=int, default=1)
    p.add_argument("--holdout-fraction", type=float, default=0.2)
    p.add_argument("--cpu-threads", type=int, default=int(os.getenv("CPU_THREADS", "2")))
    p.add_argument("--seed", type=int, default=29)
    args = p.parse_args()
    prepare_v14(
        aux_path=args.aux,
        output_path=args.output,
        seeds_path=args.seeds,
        max_seed_count=args.max_seed_count,
        generation_tokens=args.generation_tokens,
        ridge=args.ridge,
        damping=args.damping,
        switch_threshold=args.switch_threshold,
        max_corrections=args.max_corrections,
        holdout_fraction=args.holdout_fraction,
        cpu_threads=args.cpu_threads,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
