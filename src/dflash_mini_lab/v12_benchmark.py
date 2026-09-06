from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from .lfm_benchmark import RealDecodeStats, normal_decode, speculative_decode
from .lfm_dspark import LfmDSparkRuntime
from .v11_benchmark import V11_METHOD, v11_decode
from .v11_boltzmann_mobs import V11Config
from .v12_parareal import V12Config, config_dict, load_linear_model, select_v12_parareal


V12_METHOD = "parareal_linear_v12"
METHODS = ("normal", "dflash", V11_METHOD, V12_METHOD)
LABELS = {
    "normal": "Normal LFM",
    "dflash": "DFlash",
    V11_METHOD: "V11 Boltzmann-Gated MOBS",
    V12_METHOD: "V12 PARAREAL Linear",
}


def _read_prompts(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(x) for x in payload]
    for key in ("prompts", "items", "test"):
        if key in payload:
            return [str(x) for x in payload[key]]
    raise ValueError(f"{path} must contain a prompt list or a 'prompts' key")


def _greedy_reference(runtime: LfmDSparkRuntime, input_ids: np.ndarray, max_new_tokens: int) -> np.ndarray:
    seq = np.asarray(input_ids, dtype=np.int64).copy()
    for _ in range(int(max_new_tokens)):
        logits = runtime.target_logits(seq)
        seq = np.append(seq, int(np.argmax(logits[-1])))
    return seq


def _verify(runtime: LfmDSparkRuntime, seq: np.ndarray, proposal: np.ndarray):
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


def v12_decode(
    runtime: LfmDSparkRuntime,
    input_ids: np.ndarray,
    max_new_tokens: int,
    *,
    model,
    config: V12Config,
):
    seq = np.asarray(input_ids, dtype=np.int64).copy()
    start_len = int(seq.size)
    target_calls = draft_calls = accepted_total = proposed_total = 0
    target_seconds = context_seconds = draft_seconds = selection_seconds = 0.0
    linear_candidate_scores = 0
    update_rms: list[float] = []
    wall0 = time.perf_counter()

    while int(seq.size) - start_len < int(max_new_tokens):
        remaining = int(max_new_tokens) - (int(seq.size) - start_len)

        t0 = time.perf_counter()
        context = runtime.context_features(seq)
        context_seconds += time.perf_counter() - t0

        t0 = time.perf_counter()
        draft_logits = runtime.draft_logits(context)
        draft_seconds += time.perf_counter() - t0
        draft_calls += 1

        t0 = time.perf_counter()
        full, meta = select_v12_parareal(runtime, draft_logits, context, model, config)
        selection_seconds += time.perf_counter() - t0
        proposal = np.asarray(full[: min(int(full.size), remaining)], dtype=np.int64)
        proposed_total += int(proposal.size)
        linear_candidate_scores += int(meta.get("candidate_scores", 0))
        update_rms.extend(float(x) for x in meta.get("update_rms", []))

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
        method=V12_METHOD,
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
        "v12_linear_candidate_scores": int(linear_candidate_scores),
        "v12_correction_rounds": int(config.correction_rounds),
        "v12_mean_update_rms": float(np.mean(update_rms)) if update_rms else 0.0,
    }
    return seq, stats, meta


def _summary(rows: list[dict]) -> dict:
    mean = lambda key: float(statistics.fmean(float(row[key]) for row in rows))
    median = lambda key: float(statistics.median(float(row[key]) for row in rows))
    result = {
        "tokens_per_second_median": median("tokens_per_second"),
        "latency_seconds_median": median("wall_seconds"),
        "target_seconds_median": median("target_seconds"),
        "draft_seconds_median": median("draft_seconds"),
        "selection_seconds_median": median("selection_seconds"),
        "mean_target_forward_passes": mean("target_forward_passes"),
        "mean_acceptance_rate": mean("acceptance_rate"),
        "mean_tokens_per_target_pass": mean("tokens_per_target_pass"),
        "all_exact": bool(all(bool(row["exact_match"]) for row in rows)),
    }
    if any("v12_linear_candidate_scores" in row for row in rows):
        result["mean_v12_linear_candidate_scores"] = float(
            statistics.fmean(float(row.get("v12_linear_candidate_scores", 0)) for row in rows)
        )
        result["mean_v12_update_rms"] = float(
            statistics.fmean(float(row.get("v12_mean_update_rms", 0)) for row in rows)
        )
    return result


def benchmark(
    runtime: LfmDSparkRuntime,
    prompts: list[str],
    *,
    model,
    v12_config: V12Config,
    v11_config: V11Config,
    tokens: int,
    repeats: int,
) -> tuple[dict, dict]:
    rows = {method: [] for method in METHODS}

    for prompt_index, prompt in enumerate(prompts):
        ids = runtime.encode(prompt)
        reference = _greedy_reference(runtime, ids, int(tokens))
        for repeat in range(max(1, int(repeats))):
            shift = (prompt_index + repeat) % len(METHODS)
            order = METHODS[shift:] + METHODS[:shift]
            for method in order:
                if method == "normal":
                    output, stats = normal_decode(runtime, ids, int(tokens))
                    meta = {}
                elif method == "dflash":
                    output, stats = speculative_decode(
                        runtime,
                        ids,
                        int(tokens),
                        "dflash",
                        top_k=int(v12_config.top_k),
                    )
                    meta = {}
                elif method == V11_METHOD:
                    output, stats, meta = v11_decode(
                        runtime,
                        ids,
                        int(tokens),
                        config=v11_config,
                    )
                else:
                    output, stats, meta = v12_decode(
                        runtime,
                        ids,
                        int(tokens),
                        model=model,
                        config=v12_config,
                    )
                row = stats.to_dict()
                row.update(meta)
                row.update(
                    prompt=prompt,
                    prompt_index=int(prompt_index),
                    repeat=int(repeat),
                    exact_match=bool(np.array_equal(output, reference)),
                )
                rows[method].append(row)

    summary = {method: _summary(rows[method]) for method in METHODS}
    normal_tps = float(summary["normal"]["tokens_per_second_median"])
    for method in METHODS:
        summary[method]["speedup_vs_normal"] = float(
            summary[method]["tokens_per_second_median"] / max(normal_tps, 1e-12)
        )
    return rows, summary


def _markdown_report(data: dict) -> str:
    summary = data["summary"]
    lines = [
        "# DFlash12-PARAREAL benchmark",
        "",
        "DFlash12 uses a closed-form linear residual model to approximate the Parareal `F-G` correction in top-k logit space. All correction slots are evaluated with vectorized linear algebra; final output remains exact because the target verifier is authoritative.",
        "",
        "| Method | Median tok/s | vs normal | Acceptance | Mean target passes | Selection s | Exact |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for method in METHODS:
        s = summary[method]
        lines.append(
            f"| {LABELS[method]} | {s['tokens_per_second_median']:.3f} | "
            f"{s['speedup_vs_normal']:.3f}× | {100*s['mean_acceptance_rate']:.1f}% | "
            f"{s['mean_target_forward_passes']:.2f} | {s['selection_seconds_median']:.4f} | "
            f"{'✓' if s['all_exact'] else '✗'} |"
        )

    convergence = data.get("artifact_metadata", {}).get("holdout_convergence")
    if convergence:
        lines += [
            "",
            "## Teacher-space convergence recorded during regression preparation",
            "",
            "| Round | MSE | log(MSE) | Fine top-1 agreement |",
            "|---:|---:|---:|---:|",
        ]
        mse = convergence.get("teacher_mse_by_round", [])
        log_mse = convergence.get("log_teacher_mse_by_round", [])
        agree = convergence.get("fine_top1_agreement_by_round", [])
        for i in range(min(len(mse), len(log_mse), len(agree))):
            lines.append(f"| {i} | {mse[i]:.6f} | {log_mse[i]:.6f} | {100*agree[i]:.2f}% |")

    lines += [
        "",
        "## Interpretation",
        "",
        "- `round 0` is the uncorrected DFlash top-k score field.",
        "- Later rounds apply the same affine residual model in parallel to every slot/candidate.",
        "- Teacher-space MSE is a diagnostic from preparation data, not an inference-time oracle.",
        "- Exactness is established only by comparison with normal greedy target decoding.",
        "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict:
    runtime = LfmDSparkRuntime(
        args.aux,
        args.dspark,
        cpu_threads=args.cpu_threads,
    )
    model = load_linear_model(args.v12_model)
    artifact_config = dict((model.metadata or {}).get("config", {}))
    config = V12Config(
        top_k=int(args.top_k if args.top_k is not None else artifact_config.get("top_k", 8)),
        correction_rounds=int(
            args.correction_rounds
            if args.correction_rounds is not None
            else artifact_config.get("correction_rounds", 2)
        ),
        damping=float(args.damping if args.damping is not None else artifact_config.get("damping", 0.75)),
        ridge=float(artifact_config.get("ridge", 1e-3)),
        residual_clip=float(artifact_config.get("residual_clip", 6.0)),
        interpolation=tuple(float(x) for x in artifact_config.get("interpolation", (0.0, 0.5, 0.75))),
    )
    prompts = _read_prompts(args.prompts)[: max(1, int(args.max_prompts))]
    rows, summary = benchmark(
        runtime,
        prompts,
        model=model,
        v12_config=config,
        v11_config=V11Config(),
        tokens=int(args.tokens),
        repeats=int(args.repeats),
    )
    data = {
        "algorithm": "DFlash12-PARAREAL",
        "model_id": runtime.model_id,
        "prompts": len(prompts),
        "tokens_per_prompt": int(args.tokens),
        "repeats": int(args.repeats),
        "config": config_dict(config),
        "artifact_metadata": model.metadata or {},
        "summary": summary,
        "rows": rows,
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "v12_benchmark.json").write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "v12_benchmark.md").write_text(_markdown_report(data), encoding="utf-8")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DFlash12-PARAREAL against Normal, DFlash and V11")
    parser.add_argument("--aux", default="lfm-artifacts/lfm_aux.pt")
    parser.add_argument("--dspark", default="lfm-artifacts/lfm_dspark.pt")
    parser.add_argument("--v12-model", default="lfm-artifacts/v12_parareal.json")
    parser.add_argument("--prompts", default="real_benchmarks/test_prompts.json")
    parser.add_argument("--output-dir", default="v12-reports")
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-prompts", type=int, default=8)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--correction-rounds", type=int)
    parser.add_argument("--damping", type=float)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    data = run(args)
    print(json.dumps({"summary": data["summary"], "config": data["config"]}, indent=2))


if __name__ == "__main__":
    main()
