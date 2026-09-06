from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


FEATURE_NAMES = (
    "current_score",
    "coarse_score",
    "candidate_rank",
    "block_position",
    "coarse_margin",
    "candidate_last_similarity",
    "candidate_mean_similarity",
    "current_x_position",
)


@dataclass(frozen=True)
class V12Config:
    """Configuration for DFlash12-PARAREAL linear residual correction."""

    top_k: int = 8
    correction_rounds: int = 2
    damping: float = 0.75
    ridge: float = 1e-3
    residual_clip: float = 6.0
    interpolation: tuple[float, ...] = (0.0, 0.5, 0.75)


@dataclass(frozen=True)
class PararealTrainingExample:
    """One teacher block represented only by compact top-k score features."""

    coarse_scores: np.ndarray
    fine_scores: np.ndarray
    candidate_last_similarity: np.ndarray
    candidate_mean_similarity: np.ndarray


@dataclass
class PararealLinearModel:
    """Small standardized ridge-regression residual model."""

    coefficients: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    feature_names: tuple[str, ...] = FEATURE_NAMES
    metadata: dict | None = None

    def predict_residual(self, raw_features: np.ndarray, *, clip: float | None = None) -> np.ndarray:
        x = np.asarray(raw_features, dtype=np.float64)
        if x.shape[-1] != len(self.feature_names):
            raise ValueError(f"expected {len(self.feature_names)} features, got {x.shape[-1]}")
        z = (x - self.feature_mean) / self.feature_scale
        design = np.concatenate([np.ones((*z.shape[:-1], 1), dtype=np.float64), z], axis=-1)
        residual = np.tensordot(design, self.coefficients, axes=([-1], [0]))
        if clip is not None and float(clip) > 0:
            residual = np.clip(residual, -float(clip), float(clip))
        return residual.astype(np.float32, copy=False)


def _row_center(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    return values - values.mean(axis=-1, keepdims=True)


def _feature_tensor(
    current_scores: np.ndarray,
    coarse_scores: np.ndarray,
    candidate_last_similarity: np.ndarray,
    candidate_mean_similarity: np.ndarray,
) -> np.ndarray:
    """Build features for every block-position/candidate pair in parallel."""

    current = _row_center(current_scores)
    coarse = _row_center(coarse_scores)
    if current.shape != coarse.shape:
        raise ValueError("current_scores and coarse_scores must have the same shape")
    if current.ndim != 2:
        raise ValueError("scores must have shape [block, top_k]")
    block, width = current.shape
    last_sim = np.asarray(candidate_last_similarity, dtype=np.float32)
    mean_sim = np.asarray(candidate_mean_similarity, dtype=np.float32)
    if last_sim.shape != current.shape or mean_sim.shape != current.shape:
        raise ValueError("candidate similarity matrices must match score shape")

    rank = np.arange(width, dtype=np.float32) / max(width - 1, 1)
    position = np.arange(block, dtype=np.float32) / max(block - 1, 1)
    rank = np.broadcast_to(rank[None, :], current.shape)
    position = np.broadcast_to(position[:, None], current.shape)
    if width >= 2:
        margin = coarse[:, 0] - coarse[:, 1]
    else:
        margin = np.zeros(block, dtype=np.float32)
    margin = np.broadcast_to(margin[:, None], current.shape)

    return np.stack(
        [
            current,
            coarse,
            rank,
            position,
            margin,
            last_sim,
            mean_sim,
            current * position,
        ],
        axis=-1,
    ).astype(np.float32, copy=False)


def embedding_similarities(
    candidate_embeddings: np.ndarray,
    last_embedding: np.ndarray,
    mean_embedding: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Cheap semantic features: scaled candidate similarity to last/prefix-mean embeddings."""

    candidates = np.asarray(candidate_embeddings, dtype=np.float32)
    if candidates.ndim != 3:
        raise ValueError("candidate_embeddings must have shape [block, top_k, hidden]")
    hidden = int(candidates.shape[-1])
    scale = 1.0 / math.sqrt(max(hidden, 1))
    last = np.asarray(last_embedding, dtype=np.float32).reshape(hidden)
    mean = np.asarray(mean_embedding, dtype=np.float32).reshape(hidden)
    return (
        np.einsum("bkh,h->bk", candidates, last, optimize=True) * scale,
        np.einsum("bkh,h->bk", candidates, mean, optimize=True) * scale,
    )


def fit_linear_residual_model(
    examples: Iterable[PararealTrainingExample],
    *,
    config: V12Config,
    metadata: dict | None = None,
) -> PararealLinearModel:
    """Fit F-G with closed-form ridge regression.

    Intermediate states between G and F are included so the same affine
    residual model can be applied repeatedly instead of being trained only at
    iteration zero.
    """

    feature_rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    count = 0
    for example in examples:
        coarse = _row_center(example.coarse_scores)
        fine = _row_center(example.fine_scores)
        if coarse.shape != fine.shape:
            raise ValueError("coarse/fine score shapes differ")
        for tau in config.interpolation:
            t = min(0.999999, max(0.0, float(tau)))
            current = (1.0 - t) * coarse + t * fine
            features = _feature_tensor(
                current,
                coarse,
                example.candidate_last_similarity,
                example.candidate_mean_similarity,
            )
            feature_rows.append(features.reshape(-1, features.shape[-1]).astype(np.float64))
            targets.append((fine - current).reshape(-1).astype(np.float64))
        count += 1

    if not feature_rows:
        raise ValueError("at least one training example is required")
    x = np.concatenate(feature_rows, axis=0)
    y = np.concatenate(targets, axis=0)
    feature_mean = x.mean(axis=0)
    feature_scale = x.std(axis=0)
    feature_scale = np.where(feature_scale < 1e-6, 1.0, feature_scale)
    z = (x - feature_mean) / feature_scale
    design = np.concatenate([np.ones((z.shape[0], 1), dtype=np.float64), z], axis=1)
    regularizer = np.eye(design.shape[1], dtype=np.float64) * max(float(config.ridge), 0.0)
    regularizer[0, 0] = 0.0
    lhs = design.T @ design + regularizer
    rhs = design.T @ y
    coefficients = np.linalg.solve(lhs, rhs)

    fitted = PararealLinearModel(
        coefficients=coefficients,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        metadata={
            **(metadata or {}),
            "training_examples": int(count),
            "training_rows": int(x.shape[0]),
            "ridge": float(config.ridge),
            "interpolation": [float(x) for x in config.interpolation],
            "feature_names": list(FEATURE_NAMES),
        },
    )
    prediction = fitted.predict_residual(x.reshape(1, x.shape[0], x.shape[1]))[0]
    residual = y - prediction.astype(np.float64)
    fitted.metadata["train_residual_mse"] = float(np.mean(residual * residual))
    return fitted


def refine_scores(
    model: PararealLinearModel,
    coarse_scores: np.ndarray,
    candidate_last_similarity: np.ndarray,
    candidate_mean_similarity: np.ndarray,
    *,
    rounds: int,
    damping: float,
    residual_clip: float,
    fine_scores: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Apply Parareal-style affine residual corrections to all slots in parallel."""

    coarse = _row_center(coarse_scores)
    current = coarse.copy()
    fine = _row_center(fine_scores) if fine_scores is not None else None
    teacher_mse: list[float] = []
    update_rms: list[float] = []
    if fine is not None:
        teacher_mse.append(float(np.mean((fine - current) ** 2)))

    for _ in range(max(0, int(rounds))):
        features = _feature_tensor(
            current,
            coarse,
            candidate_last_similarity,
            candidate_mean_similarity,
        )
        delta = model.predict_residual(features, clip=residual_clip)
        update = float(damping) * delta
        current = _row_center(current + update)
        update_rms.append(float(np.sqrt(np.mean(update.astype(np.float64) ** 2))))
        if fine is not None:
            teacher_mse.append(float(np.mean((fine - current) ** 2)))

    diagnostics: dict = {"update_rms": update_rms}
    if teacher_mse:
        eps = 1e-12
        log_error = [float(math.log(max(x, eps))) for x in teacher_mse]
        contraction = [
            float(teacher_mse[i + 1] / max(teacher_mse[i], eps))
            for i in range(len(teacher_mse) - 1)
        ]
        diagnostics.update(
            teacher_mse=teacher_mse,
            log_teacher_mse=log_error,
            contraction_ratio=contraction,
        )
    return current, diagnostics


def evaluate_convergence(
    model: PararealLinearModel,
    examples: Iterable[PararealTrainingExample],
    *,
    config: V12Config,
) -> dict:
    """Aggregate teacher-space error by correction round."""

    per_round: list[list[float]] = [[] for _ in range(max(0, int(config.correction_rounds)) + 1)]
    top1_match: list[list[float]] = [[] for _ in per_round]
    count = 0
    for example in examples:
        coarse = _row_center(example.coarse_scores)
        fine = _row_center(example.fine_scores)
        states = [coarse]
        current = coarse.copy()
        for _ in range(max(0, int(config.correction_rounds))):
            features = _feature_tensor(
                current,
                coarse,
                example.candidate_last_similarity,
                example.candidate_mean_similarity,
            )
            current = _row_center(
                current + float(config.damping) * model.predict_residual(features, clip=config.residual_clip)
            )
            states.append(current)
        for i, state in enumerate(states):
            per_round[i].append(float(np.mean((fine - state) ** 2)))
            top1_match[i].append(float(np.mean(np.argmax(state, axis=-1) == np.argmax(fine, axis=-1))))
        count += 1

    if count == 0:
        raise ValueError("at least one evaluation example is required")
    mse = [float(np.mean(values)) for values in per_round]
    eps = 1e-12
    return {
        "examples": int(count),
        "teacher_mse_by_round": mse,
        "log_teacher_mse_by_round": [float(math.log(max(x, eps))) for x in mse],
        "contraction_ratio_by_round": [
            float(mse[i + 1] / max(mse[i], eps)) for i in range(len(mse) - 1)
        ],
        "fine_top1_agreement_by_round": [float(np.mean(values)) for values in top1_match],
    }


def select_v12_parareal(runtime, draft_logits: np.ndarray, context: np.ndarray, model: PararealLinearModel, config: V12Config):
    """Vectorized top-k Parareal correction used by the real LFM benchmark."""

    _, top_ids, top_vals = runtime._top_k(draft_logits, int(config.top_k))
    if top_ids.size == 0:
        return np.empty(0, dtype=np.int64), {
            "candidate_scores": 0,
            "correction_rounds": int(config.correction_rounds),
            "update_rms": [],
        }

    import torch

    index = torch.from_numpy(np.asarray(top_ids, dtype=np.int64)).long()
    candidate_embeddings = runtime.embedding[index].detach().float().cpu().numpy()
    hidden = int(candidate_embeddings.shape[-1])
    context_tokens = int(runtime.config.context_tokens)
    last_start = (context_tokens - 1) * hidden
    last_embedding = np.asarray(context[last_start : last_start + hidden], dtype=np.float32)
    mean_start = context_tokens * hidden
    mean_embedding = np.asarray(context[mean_start : mean_start + hidden], dtype=np.float32)
    last_sim, mean_sim = embedding_similarities(candidate_embeddings, last_embedding, mean_embedding)

    refined, diagnostics = refine_scores(
        model,
        top_vals,
        last_sim,
        mean_sim,
        rounds=config.correction_rounds,
        damping=config.damping,
        residual_clip=config.residual_clip,
    )
    best = np.argmax(refined, axis=-1).astype(np.int64)
    proposal = top_ids[np.arange(top_ids.shape[0]), best].astype(np.int64, copy=False)
    diagnostics.update(
        candidate_scores=int(top_ids.size) * max(0, int(config.correction_rounds)),
        correction_rounds=int(config.correction_rounds),
        top_k=int(config.top_k),
    )
    return proposal, diagnostics


def save_linear_model(path: str | Path, model: PararealLinearModel) -> None:
    payload = {
        "format_version": 1,
        "algorithm": "DFlash12-PARAREAL",
        "feature_names": list(model.feature_names),
        "coefficients": np.asarray(model.coefficients, dtype=np.float64).tolist(),
        "feature_mean": np.asarray(model.feature_mean, dtype=np.float64).tolist(),
        "feature_scale": np.asarray(model.feature_scale, dtype=np.float64).tolist(),
        "metadata": model.metadata or {},
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_linear_model(path: str | Path) -> PararealLinearModel:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("algorithm") != "DFlash12-PARAREAL":
        raise ValueError("not a DFlash12-PARAREAL model")
    names = tuple(str(x) for x in payload["feature_names"])
    if names != FEATURE_NAMES:
        raise ValueError(f"feature contract mismatch: {names}")
    return PararealLinearModel(
        coefficients=np.asarray(payload["coefficients"], dtype=np.float64),
        feature_mean=np.asarray(payload["feature_mean"], dtype=np.float64),
        feature_scale=np.asarray(payload["feature_scale"], dtype=np.float64),
        feature_names=names,
        metadata=dict(payload.get("metadata", {})),
    )


def config_dict(config: V12Config) -> dict:
    result = asdict(config)
    result["interpolation"] = [float(x) for x in config.interpolation]
    return result
