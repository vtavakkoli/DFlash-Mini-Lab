from __future__ import annotations

import numpy as np

from dflash_mini_lab.v12_parareal import (
    PararealTrainingExample,
    V12Config,
    evaluate_convergence,
    fit_linear_residual_model,
    load_linear_model,
    refine_scores,
    save_linear_model,
)


def _synthetic_examples(count: int = 32) -> list[PararealTrainingExample]:
    rng = np.random.default_rng(17)
    examples: list[PararealTrainingExample] = []
    rank = np.arange(8, dtype=np.float32)[None, :] / 7.0
    position = np.arange(4, dtype=np.float32)[:, None] / 3.0
    for _ in range(count):
        coarse = rng.normal(size=(4, 8)).astype(np.float32)
        last = rng.normal(scale=0.2, size=(4, 8)).astype(np.float32)
        mean = rng.normal(scale=0.2, size=(4, 8)).astype(np.float32)
        fine = coarse + (
            -0.45 * coarse
            + 0.25 * last
            + 0.15 * mean
            - 0.10 * rank
            + 0.07 * position
        ).astype(np.float32)
        examples.append(
            PararealTrainingExample(
                coarse_scores=coarse,
                fine_scores=fine,
                candidate_last_similarity=last,
                candidate_mean_similarity=mean,
            )
        )
    return examples


def test_v12_linear_parareal_converges_geometrically_on_affine_problem():
    examples = _synthetic_examples()
    config = V12Config(correction_rounds=2, damping=0.75, ridge=1e-6)
    model = fit_linear_residual_model(examples[:24], config=config)
    metrics = evaluate_convergence(model, examples[24:], config=config)

    mse = metrics["teacher_mse_by_round"]
    assert len(mse) == 3
    assert mse[1] < mse[0]
    assert mse[2] < mse[1]
    assert all(ratio < 0.25 for ratio in metrics["contraction_ratio_by_round"])
    assert metrics["fine_top1_agreement_by_round"][-1] >= metrics["fine_top1_agreement_by_round"][0]


def test_v12_refinement_is_parallel_shape_preserving_and_deterministic():
    examples = _synthetic_examples(12)
    config = V12Config(correction_rounds=2, damping=0.75, ridge=1e-6)
    model = fit_linear_residual_model(examples[:8], config=config)
    example = examples[8]

    first, diagnostics_a = refine_scores(
        model,
        example.coarse_scores,
        example.candidate_last_similarity,
        example.candidate_mean_similarity,
        rounds=2,
        damping=0.75,
        residual_clip=6.0,
        fine_scores=example.fine_scores,
    )
    second, diagnostics_b = refine_scores(
        model,
        example.coarse_scores,
        example.candidate_last_similarity,
        example.candidate_mean_similarity,
        rounds=2,
        damping=0.75,
        residual_clip=6.0,
        fine_scores=example.fine_scores,
    )

    assert first.shape == (4, 8)
    assert np.array_equal(first, second)
    assert diagnostics_a == diagnostics_b
    assert len(diagnostics_a["update_rms"]) == 2


def test_v12_json_artifact_round_trip(tmp_path):
    examples = _synthetic_examples(8)
    config = V12Config(correction_rounds=2, damping=0.75, ridge=1e-6)
    model = fit_linear_residual_model(
        examples,
        config=config,
        metadata={"model_id": "synthetic/test"},
    )
    path = tmp_path / "v12_parareal.json"
    save_linear_model(path, model)
    restored = load_linear_model(path)

    assert restored.feature_names == model.feature_names
    assert np.allclose(restored.coefficients, model.coefficients)
    assert np.allclose(restored.feature_mean, model.feature_mean)
    assert np.allclose(restored.feature_scale, model.feature_scale)
    assert restored.metadata["model_id"] == "synthetic/test"
