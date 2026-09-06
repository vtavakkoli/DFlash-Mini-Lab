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
from .v12_parareal import (
    PararealTrainingExample,
    V12Config,
    config_dict,
    embedding_similarities,
    evaluate_convergence,
    fit_linear_residual_model,
    save_linear_model,
)


def _greedy_trajectory(runtime: LfmReferenceRuntime, prompt_ids: np.ndarray, new_tokens: int) -> np.ndarray:
    seq = np.asarray(prompt_ids, dtype=np.int64).copy()
    for _ in range(max(1, int(new_tokens))):
        logits = runtime.target_logits(seq)
        seq = np.append(seq, int(np.argmax(logits[-1])))
    return seq


def _context_embeddings(runtime: LfmReferenceRuntime, context: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hidden = int(runtime.config.target_hidden_size)
    context_tokens = int(runtime.config.context_tokens)
    last_start = (context_tokens - 1) * hidden
    mean_start = context_tokens * hidden
    return (
        np.asarray(context[last_start : last_start + hidden], dtype=np.float32),
        np.asarray(context[mean_start : mean_start + hidden], dtype=np.float32),
    )


def collect_regression_examples(
    runtime: LfmReferenceRuntime,
    prompts: list[str],
    *,
    generation_tokens: int,
    top_k: int,
    stride: int = 1,
) -> list[PararealTrainingExample]:
    """Build teacher/coarse top-k score pairs without per-window target forwards.

    A greedy trajectory is generated first. Because the target is causal, one
    full teacher forward over that trajectory supplies the fine logits for every
    block window on the same sequence.
    """

    examples: list[PararealTrainingExample] = []
    block = int(runtime.block_size)
    for prompt in prompts:
        prompt_ids = runtime.encode(prompt)
        trajectory = _greedy_trajectory(runtime, prompt_ids, int(generation_tokens) + block)
        teacher_logits = runtime.target_logits(trajectory)
        prompt_end = int(prompt_ids.size) - 1
        last_end = int(trajectory.size) - block - 1

        for end in range(prompt_end, last_end + 1, max(1, int(stride))):
            prefix = trajectory[: end + 1]
            context = runtime.context_features(prefix)
            draft_logits = runtime.draft_logits(context)
            _, top_ids, top_vals = runtime._top_k(draft_logits, int(top_k))

            # logit[end + j] predicts token end + j + 1, i.e. future slot j.
            fine_full = teacher_logits[end : end + block]
            fine_vals = np.take_along_axis(fine_full, top_ids, axis=1).astype(np.float32, copy=False)

            index = torch.from_numpy(np.asarray(top_ids, dtype=np.int64)).long()
            candidate_embeddings = runtime.embedding[index].detach().float().cpu().numpy()
            last_embedding, mean_embedding = _context_embeddings(runtime, context)
            last_sim, mean_sim = embedding_similarities(
                candidate_embeddings,
                last_embedding,
                mean_embedding,
            )
            examples.append(
                PararealTrainingExample(
                    coarse_scores=np.asarray(top_vals, dtype=np.float32),
                    fine_scores=fine_vals,
                    candidate_last_similarity=last_sim,
                    candidate_mean_similarity=mean_sim,
                )
            )
    if not examples:
        raise RuntimeError("no DFlash12 regression examples were constructed")
    return examples


def prepare_v12(
    *,
    aux_path: str | Path,
    output_path: str | Path,
    seeds_path: str | Path,
    max_seed_count: int = 24,
    generation_tokens: int = 24,
    top_k: int = 8,
    correction_rounds: int = 2,
    damping: float = 0.75,
    ridge: float = 1e-3,
    residual_clip: float = 6.0,
    stride: int = 1,
    holdout_fraction: float = 0.2,
    cpu_threads: int = 2,
    seed: int = 23,
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
    config = V12Config(
        top_k=int(top_k),
        correction_rounds=int(correction_rounds),
        damping=float(damping),
        ridge=float(ridge),
        residual_clip=float(residual_clip),
    )
    examples = collect_regression_examples(
        runtime,
        prompts,
        generation_tokens=int(generation_tokens),
        top_k=int(top_k),
        stride=int(stride),
    )

    split = int(round(len(examples) * (1.0 - float(holdout_fraction))))
    split = min(max(1, split), len(examples) - 1)
    train_examples = examples[:split]
    holdout_examples = examples[split:]
    metadata = {
        "algorithm": "DFlash12-PARAREAL",
        "mechanism": "parallel top-k logit residual correction using closed-form ridge regression",
        "parareal_mapping": {
            "coarse_G": "DFlash top-k block logits",
            "fine_F_training_only": "teacher target logits on the greedy trajectory",
            "learned_residual": "linear surrogate of F-G",
            "inference_update": "q_{k+1}=center(q_k+damping*R_linear(q_k,G,features))",
        },
        "model_id": runtime.model_id,
        "candidate_size": int(runtime.candidate_size),
        "block_size": int(runtime.block_size),
        "target_weights_redistributed": False,
        "seed_count": len(prompts),
        "generation_tokens_per_seed": int(generation_tokens),
        "config": config_dict(config),
    }
    model = fit_linear_residual_model(train_examples, config=config, metadata=metadata)
    train_convergence = evaluate_convergence(model, train_examples, config=config)
    holdout_convergence = evaluate_convergence(model, holdout_examples, config=config)
    model.metadata.update(
        train_examples=len(train_examples),
        holdout_examples=len(holdout_examples),
        train_convergence=train_convergence,
        holdout_convergence=holdout_convergence,
    )
    save_linear_model(output_path, model)

    result = {
        **metadata,
        "artifact": str(output_path),
        "training_examples": len(train_examples),
        "holdout_examples": len(holdout_examples),
        "train_convergence": train_convergence,
        "holdout_convergence": holdout_convergence,
        "coefficients": {
            "intercept": float(model.coefficients[0]),
            **{
                name: float(value)
                for name, value in zip(model.feature_names, model.coefficients[1:])
            },
        },
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit the DFlash12-PARAREAL linear residual corrector on frozen LFM teacher trajectories"
    )
    parser.add_argument("--aux", default="lfm-artifacts/lfm_aux.pt")
    parser.add_argument("--output", default="lfm-artifacts/v12_parareal.json")
    parser.add_argument("--seeds", default="real_benchmarks/train_seeds.json")
    parser.add_argument("--max-seed-count", type=int, default=24)
    parser.add_argument("--generation-tokens", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--correction-rounds", type=int, default=2)
    parser.add_argument("--damping", type=float, default=0.75)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--residual-clip", type=float, default=6.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--cpu-threads", type=int, default=int(os.getenv("CPU_THREADS", "2")))
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    prepare_v12(
        aux_path=args.aux,
        output_path=args.output,
        seeds_path=args.seeds,
        max_seed_count=args.max_seed_count,
        generation_tokens=args.generation_tokens,
        top_k=args.top_k,
        correction_rounds=args.correction_rounds,
        damping=args.damping,
        ridge=args.ridge,
        residual_clip=args.residual_clip,
        stride=args.stride,
        holdout_fraction=args.holdout_fraction,
        cpu_threads=args.cpu_threads,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
