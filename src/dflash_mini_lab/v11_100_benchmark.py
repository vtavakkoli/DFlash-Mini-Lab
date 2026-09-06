from __future__ import annotations

import argparse
import html
import json
import statistics
from pathlib import Path

import numpy as np

from .lfm_benchmark import _read_prompts, normal_decode, speculative_decode
from .lfm_dspark import LfmDSparkRuntime
from .lfm_v10 import V10Config
from .lfm_v9v10_benchmark import _greedy_reference, dspark_decode, v10_decode
from .v11_benchmark import V11_METHOD, v11_decode
from .v11_boltzmann_mobs import V11Config


METHODS = (
    "normal",
    "dflash",
    "dflash3_mobs",
    "dflash6_boltzmann",
    "dspark_v9",
    "boltzmann_v10",
    V11_METHOD,
)
LABELS = {
    "normal": "Normal LFM",
    "dflash": "DFlash",
    "dflash3_mobs": "DFlash3-MOBS",
    "dflash6_boltzmann": "DFlash6-Boltzmann",
    "dspark_v9": "V9 DSpark-Lite",
    "boltzmann_v10": "V10 Advanced Boltzmann",
    V11_METHOD: "V11 Boltzmann-Gated MOBS",
}

# Frozen from the prior held-out 4-prompt calibration. The 100 evaluation prompts
# never influence these settings.
DSPARK_FLOOR = 0.0
V10_CONFIG = V10Config(
    temperature=0.03,
    margin_cutoff=0.15,
    stochastic_budget=1,
    margin_slope=1.5,
)
V11_CONFIG = V11Config(
    temperature=0.04,
    gate_floor=0.08,
    mobs_budget=3,
    top_k=4,
    pair_weight=1.0,
    early_position_bias=0.15,
)


def _run_method(runtime, method: str, ids: np.ndarray, tokens: int, top_k: int):
    if method == "normal":
        out, stats = normal_decode(runtime, ids, tokens)
        return out, stats, {}
    if method == "dflash":
        out, stats = speculative_decode(runtime, ids, tokens, "dflash", top_k=top_k)
        return out, stats, {}
    if method == "dflash3_mobs":
        out, stats = speculative_decode(runtime, ids, tokens, "dflash3_mobs", top_k=top_k)
        return out, stats, {}
    if method == "dflash6_boltzmann":
        out, stats = speculative_decode(
            runtime,
            ids,
            tokens,
            "dflash6_boltzmann",
            top_k=top_k,
            boltzmann_temperature=0.15,
        )
        return out, stats, {}
    if method == "dspark_v9":
        return dspark_decode(
            runtime,
            ids,
            tokens,
            top_k=top_k,
            survival_floor=DSPARK_FLOOR,
        )
    if method == "boltzmann_v10":
        return v10_decode(runtime, ids, tokens, config=V10_CONFIG)
    if method == V11_METHOD:
        return v11_decode(runtime, ids, tokens, config=V11_CONFIG)
    raise ValueError(method)


def _summary(rows: list[dict], tokens: int) -> dict:
    walls = np.asarray([float(x["wall_seconds"]) for x in rows], dtype=np.float64)
    total_tokens = int(tokens) * len(rows)
    return {
        "aggregate_tokens_per_second": float(total_tokens / max(float(walls.sum()), 1e-12)),
        "tokens_per_second_median": float(statistics.median(float(x["tokens_per_second"]) for x in rows)),
        "wall_seconds_total": float(walls.sum()),
        "wall_seconds_median": float(np.median(walls)),
        "mean_acceptance_rate": float(statistics.fmean(float(x["acceptance_rate"]) for x in rows)),
        "mean_target_forward_passes": float(statistics.fmean(float(x["target_forward_passes"]) for x in rows)),
        "mean_tokens_per_target_pass": float(statistics.fmean(float(x["tokens_per_target_pass"]) for x in rows)),
        "mean_total_guidance_scores": float(statistics.fmean(float(x["total_guidance_scores"]) for x in rows)),
        "all_exact": bool(all(bool(x["exact_match"]) for x in rows)),
    }


def _bootstrap_ratio(
    base_walls: np.ndarray,
    method_walls: np.ndarray,
    *,
    samples: int = 5000,
    seed: int = 1100,
) -> dict:
    if base_walls.shape != method_walls.shape:
        raise ValueError("paired arrays must have identical shape")
    n = int(base_walls.size)
    point = float(base_walls.sum() / max(float(method_walls.sum()), 1e-12))
    rng = np.random.default_rng(seed)
    ratios = np.empty(int(samples), dtype=np.float64)
    for i in range(int(samples)):
        idx = rng.integers(0, n, size=n)
        ratios[i] = float(base_walls[idx].sum() / max(float(method_walls[idx].sum()), 1e-12))
    lo, hi = np.quantile(ratios, [0.025, 0.975])
    return {
        "point": point,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "bootstrap_samples": int(samples),
        "seed": int(seed),
    }


def _build_report(data: dict, output: str | Path) -> None:
    summary = data["summary"]
    ordered = sorted(METHODS, key=lambda m: summary[m]["aggregate_tokens_per_second"], reverse=True)
    rows = []
    for method in ordered:
        s = summary[method]
        ci = s["speedup_vs_dflash_ci95"]
        rows.append(
            f"<tr class='{'winner' if method == data['winner'] else ''}'>"
            f"<th>{html.escape(LABELS[method])}</th>"
            f"<td>{s['aggregate_tokens_per_second']:.3f}</td>"
            f"<td>{s['speedup_vs_normal']:.3f}×</td>"
            f"<td>{s['speedup_vs_dflash']:.3f}×</td>"
            f"<td>[{ci[0]:.3f}, {ci[1]:.3f}]</td>"
            f"<td>{100*s['mean_acceptance_rate']:.1f}%</td>"
            f"<td>{s['mean_total_guidance_scores']:.1f}</td>"
            f"<td>{'✓' if s['all_exact'] else '✗'}</td></tr>"
        )
    css = """
    :root{font-family:Inter,system-ui;background:#07111f;color:#eef7ff;--line:#294963;--cyan:#58d9ff;--green:#7cf5b2;--muted:#98abc0}
    *{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% 0,#173b5a,transparent 35%),#07111f}
    main{max-width:1180px;margin:auto;padding:30px 20px 70px}.hero,.card{background:linear-gradient(145deg,#10253b,#0a1727);border:1px solid var(--line);border-radius:22px;padding:24px;margin-bottom:20px}
    h1{font-size:clamp(2.2rem,5vw,4.3rem);line-height:1;margin:.2em 0}.chips{display:flex;gap:8px;flex-wrap:wrap}.chip{border:1px solid #355a78;border-radius:999px;padding:6px 10px}
    .grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.metric{border:1px solid var(--line);padding:14px;border-radius:15px}.metric b{display:block;font-size:1.5rem;color:var(--green)}
    .muted{color:var(--muted)}.table{overflow:auto}table{width:100%;border-collapse:collapse;min-width:900px}th,td{padding:10px;border-bottom:1px solid #20384f;text-align:right}th:first-child{text-align:left}.winner{background:#123a29}
    @media(max-width:800px){.grid{grid-template-columns:1fr 1fr}}
    """
    win = data["winner"]
    dflash_ci = summary[win]["speedup_vs_dflash_ci95"]
    verdict = (
        "Statistically clear over DFlash"
        if win != "dflash" and dflash_ci[0] > 1.0
        else "No statistically clear method beats DFlash"
    )
    doc = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
    <title>100-prompt LFM V11 validation</title><style>{css}</style></head><body><main>
    <section class='hero'><div class='chips'><span class='chip'>100 held-out prompts</span><span class='chip'>24 tokens each</span><span class='chip'>float32 CPU · 2 threads</span><span class='chip'>frozen calibration</span></div>
    <h1>100-prompt validation</h1><p class='muted'>A broad prompt-distribution test of DFlash, MOBS, Boltzmann, DSpark and V11. Algorithm settings were frozen before these 100 prompts were introduced.</p>
    <div class='grid'><div class='metric'><small>Aggregate winner</small><b>{html.escape(LABELS[win])}</b></div>
    <div class='metric'><small>Winner throughput</small><b>{summary[win]['aggregate_tokens_per_second']:.3f} tok/s</b></div>
    <div class='metric'><small>vs Normal</small><b>{summary[win]['speedup_vs_normal']:.3f}×</b></div>
    <div class='metric'><small>Inference</small><b>{html.escape(verdict)}</b></div></div></section>
    <section class='card'><h2>Measured ranking</h2><div class='table'><table><thead><tr><th>Method</th><th>aggregate tok/s</th><th>vs Normal</th><th>vs DFlash</th><th>95% CI vs DFlash</th><th>Acceptance</th><th>Guidance ops</th><th>Exact</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
    <section class='card'><h2>Methodology</h2><p>100 unique prompts cover factual completion, reasoning, programming, systems, AI/ML, mathematics, science/engineering, urban infrastructure, writing and everyday common sense. None are used by the 40-seed distillation set or the 4-prompt calibration set. Each method generates 24 tokens once per prompt. Method order rotates across prompts to reduce thermal/turbo/order bias. Reported speed uses total generated tokens divided by total wall time. Paired 5,000-sample bootstrap confidence intervals resample prompts.</p></section>
    <section class='card'><h2>Frozen settings</h2><pre>{html.escape(json.dumps(data['frozen_settings'], indent=2))}</pre></section>
    </main></body></html>"""
    Path(output).write_text(doc, encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    prompts = _read_prompts(args.prompts, args.prompt_limit)
    if len(prompts) != int(args.prompt_limit):
        raise ValueError(f"expected {args.prompt_limit} prompts, got {len(prompts)}")
    if len(set(prompts)) != len(prompts):
        raise ValueError("evaluation prompts must be unique")

    runtime = LfmDSparkRuntime(
        args.aux,
        args.dspark,
        model_id=args.model_id,
        cpu_threads=args.cpu_threads,
        dtype=args.dtype,
    )
    warm = runtime.encode(prompts[0])
    _ = runtime.target_logits(warm)
    _ = runtime.draft_logits(runtime.context_features(warm))

    rows = {m: [] for m in METHODS}
    for pi, prompt in enumerate(prompts):
        ids = runtime.encode(prompt)
        reference = _greedy_reference(runtime, ids, args.tokens)
        shift = pi % len(METHODS)
        order = METHODS[shift:] + METHODS[:shift]
        for method in order:
            out, stats, meta = _run_method(runtime, method, ids, args.tokens, args.top_k)
            row = stats.to_dict()
            row.update(meta)
            row.update(
                prompt=prompt,
                prompt_index=pi,
                exact_match=bool(np.array_equal(out, reference)),
            )
            rows[method].append(row)

    summary = {m: _summary(rows[m], args.tokens) for m in METHODS}
    normal_walls = np.asarray([x["wall_seconds"] for x in rows["normal"]], dtype=np.float64)
    dflash_walls = np.asarray([x["wall_seconds"] for x in rows["dflash"]], dtype=np.float64)
    for i, method in enumerate(METHODS):
        walls = np.asarray([x["wall_seconds"] for x in rows[method]], dtype=np.float64)
        vs_normal = _bootstrap_ratio(normal_walls, walls, seed=1100 + i)
        vs_dflash = _bootstrap_ratio(dflash_walls, walls, seed=2100 + i)
        summary[method]["speedup_vs_normal"] = vs_normal["point"]
        summary[method]["speedup_vs_normal_ci95"] = [vs_normal["ci95_low"], vs_normal["ci95_high"]]
        summary[method]["speedup_vs_dflash"] = vs_dflash["point"]
        summary[method]["speedup_vs_dflash_ci95"] = [vs_dflash["ci95_low"], vs_dflash["ci95_high"]]

    winner = max(METHODS, key=lambda m: summary[m]["aggregate_tokens_per_second"])
    ordered = sorted(METHODS, key=lambda m: summary[m]["aggregate_tokens_per_second"], reverse=True)
    second = ordered[1]
    winner_vs_second = _bootstrap_ratio(
        np.asarray([x["wall_seconds"] for x in rows[second]], dtype=np.float64),
        np.asarray([x["wall_seconds"] for x in rows[winner]], dtype=np.float64),
        seed=3100,
    )

    payload = {
        "schema_version": 1,
        "benchmark_name": "LFM2.5-350M 100-prompt V11 validation",
        "model": {
            "id": runtime.model_id,
            "target_parameter_count": runtime.target_parameter_count,
            "candidate_size": runtime.candidate_size,
        },
        "config": {
            "prompt_count": len(prompts),
            "tokens_per_prompt": int(args.tokens),
            "repeats": 1,
            "cpu_threads": int(args.cpu_threads),
            "dtype": args.dtype,
            "execution_order": "cyclic rotation by prompt index",
            "statistical_test": "paired prompt bootstrap, 5000 samples",
        },
        "evaluation_set": {
            "path": args.prompts,
            "held_out_from_training": True,
            "held_out_from_calibration": True,
            "unique_prompt_count": len(set(prompts)),
        },
        "frozen_settings": {
            "dspark_survival_floor": DSPARK_FLOOR,
            "v10": {
                "temperature": V10_CONFIG.temperature,
                "margin_cutoff": V10_CONFIG.margin_cutoff,
                "stochastic_budget": V10_CONFIG.stochastic_budget,
                "margin_slope": V10_CONFIG.margin_slope,
            },
            "v11": {
                "temperature": V11_CONFIG.temperature,
                "gate_floor": V11_CONFIG.gate_floor,
                "mobs_budget": V11_CONFIG.mobs_budget,
                "top_k": V11_CONFIG.top_k,
                "pair_weight": V11_CONFIG.pair_weight,
                "early_position_bias": V11_CONFIG.early_position_bias,
            },
            "legacy_boltzmann_temperature": 0.15,
            "top_k": int(args.top_k),
        },
        "winner": winner,
        "runner_up": second,
        "winner_vs_runner_up": winner_vs_second,
        "clear_winner_95pct": bool(winner_vs_second["ci95_low"] > 1.0),
        "summary": summary,
        "runs": [row for method in METHODS for row in rows[method]],
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _build_report(payload, out / "report.html")
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description="100-prompt frozen-setting V11 validation")
    p.add_argument("--aux", default="lfm-artifacts/lfm_aux.pt")
    p.add_argument("--dspark", default="lfm-artifacts/lfm_dspark.pt")
    p.add_argument("--prompts", default="real_benchmarks/prompts_100.json")
    p.add_argument("--output-dir", default="v11-100-reports")
    p.add_argument("--model-id", default="LiquidAI/LFM2.5-350M-Base")
    p.add_argument("--tokens", type=int, default=24)
    p.add_argument("--prompt-limit", type=int, default=100)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--dtype", choices=("float32",), default="float32")
    args = p.parse_args()
    d = run(args)
    print(json.dumps({
        "winner": d["winner"],
        "runner_up": d["runner_up"],
        "clear_winner_95pct": d["clear_winner_95pct"],
        "winner_vs_runner_up": d["winner_vs_runner_up"],
        "summary": d["summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
