from __future__ import annotations

import argparse
import html
import json
import statistics
from pathlib import Path

import numpy as np

from . import lfm_all12_benchmark as base
from .lfm_benchmark import _read_prompts
from .lfm_dspark import LfmDSparkRuntime
from .lfm_tuning import calibrate_verify_widths
from .lfm_showcase import calibrate_act
from .lfm_v9v10_benchmark import calibrate_dspark, calibrate_v10
from .v11_benchmark import V11_METHOD, calibrate_v11
from .v12_benchmark import V12_METHOD
from .v12_parareal import load_linear_model
from .v13_minop import V13_METHOD, v13_decode
from .v14_simple_parareal import V14_METHOD, load_estimator, v14_decode


METHODS = tuple(base.METHODS) + (V13_METHOD, V14_METHOD)
METHOD_INFO = {
    **base.METHOD_INFO,
    V13_METHOD: {
        "index": 13,
        "label": "V13 MinOp",
        "short": "Fused top-2 V10 fast path",
        "mechanism": "Preserve V10's sparse two-way decision, but select top-2 inside the Torch drafter and route at most one uncertain slot.",
        "complexity": "O(B) routing after fused top-2",
        "stages": ["FUSED TOP-2", "ONE SLOT", "VERIFY", "COMMIT"],
    },
    V14_METHOD: {
        "index": 14,
        "label": "V14 Simple PARAREAL",
        "short": "Scalar fine-gap estimator",
        "mechanism": "Estimate the target fine top1-top2 gap with three affine terms, apply one Parareal residual update, and allow at most one top-2 switch.",
        "complexity": "O(B) scalar estimator",
        "stages": ["COARSE GAP", "F-HAT", "VERIFY", "COMMIT"],
    },
}


def _greedy_reference(runtime: LfmDSparkRuntime, ids: np.ndarray, tokens: int) -> np.ndarray:
    return base._greedy_reference(runtime, ids, tokens)


def _summary(rows: list[dict]) -> dict:
    mean = lambda key, default=0.0: float(statistics.fmean(float(row.get(key, default)) for row in rows))
    median = lambda key, default=0.0: float(statistics.median(float(row.get(key, default)) for row in rows))
    guidance = [
        float(row.get("total_guidance_scores", 0.0))
        + float(row.get("v12_linear_candidate_scores", 0.0))
        + float(row.get("v14_estimator_ops", 0.0))
        for row in rows
    ]
    result = {
        "tokens_per_second_median": median("tokens_per_second"),
        "latency_seconds_median": median("wall_seconds"),
        "target_seconds_median": median("target_seconds"),
        "draft_seconds_median": median("draft_seconds"),
        "selection_seconds_median": median("selection_seconds"),
        "mean_target_forward_passes": mean("target_forward_passes"),
        "mean_draft_forward_passes": mean("draft_forward_passes"),
        "mean_acceptance_rate": mean("acceptance_rate"),
        "mean_tokens_per_target_pass": mean("tokens_per_target_pass"),
        "mean_guidance_work": float(statistics.fmean(guidance)),
        "all_exact": bool(all(bool(row["exact_match"]) for row in rows)),
    }
    extras = (
        "act_mean_verified_drafts_per_block",
        "act_trimmed_draft_tokens",
        "dspark_mean_confidence",
        "v10_fast_argmax_positions",
        "v11_fast_argmax_positions",
        "v12_mean_update_rms",
        "v13_candidate_scores",
        "v13_routed_positions",
        "v13_fast_argmax_positions",
        "v13_top2_values_transferred",
        "v14_estimator_ops",
        "v14_switched_positions",
        "v14_fast_argmax_positions",
        "v14_mean_abs_residual",
    )
    for key in extras:
        if any(key in row for row in rows):
            result[key] = mean(key)
    return result


def _run_method(method, runtime, ids, tokens, **kwargs):
    previous = getattr(runtime, "max_verify_tokens", 0)
    runtime.max_verify_tokens = getattr(runtime, "verify_widths", {}).get(method, previous)
    try:
        return _run_method_impl(method, runtime, ids, tokens, **kwargs)
    finally:
        runtime.max_verify_tokens = previous


def _run_method_impl(
    method: str,
    runtime: LfmDSparkRuntime,
    ids: np.ndarray,
    tokens: int,
    *,
    top_k: int,
    jump_weight: float,
    fused_weight: float,
    boltzmann_temperature: float,
    bmobs_temperature: float,
    act_threshold: float,
    dspark_floor: float,
    v10_config,
    v11_config,
    v12_model,
    v12_config,
    v14_estimator,
):
    if method == V13_METHOD:
        return v13_decode(runtime, ids, tokens, config=v10_config)
    if method == V14_METHOD:
        return v14_decode(runtime, ids, tokens, estimator=v14_estimator)
    return base._run_method(
        method,
        runtime,
        ids,
        tokens,
        top_k=top_k,
        jump_weight=jump_weight,
        fused_weight=fused_weight,
        boltzmann_temperature=boltzmann_temperature,
        bmobs_temperature=bmobs_temperature,
        act_threshold=act_threshold,
        dspark_floor=dspark_floor,
        v10_config=v10_config,
        v11_config=v11_config,
        v12_model=v12_model,
        v12_config=v12_config,
    )


def benchmark(
    runtime: LfmDSparkRuntime,
    prompts: list[str],
    *,
    tokens: int,
    repeats: int,
    top_k: int,
    jump_weight: float,
    fused_weight: float,
    boltzmann_temperature: float,
    bmobs_temperature: float,
    act_threshold: float,
    dspark_floor: float,
    v10_config,
    v11_config,
    v12_model,
    v12_config,
    v14_estimator,
):
    rows = {method: [] for method in METHODS}
    examples = []
    for prompt_index, prompt in enumerate(prompts):
        ids = runtime.encode(prompt)
        reference = _greedy_reference(runtime, ids, tokens)
        first_text = {}
        for repeat in range(max(1, int(repeats))):
            shift = (prompt_index * max(1, int(repeats)) + repeat) % len(METHODS)
            order = METHODS[shift:] + METHODS[:shift]
            for method in order:
                output, stats, meta = _run_method(
                    method,
                    runtime,
                    ids,
                    tokens,
                    top_k=top_k,
                    jump_weight=jump_weight,
                    fused_weight=fused_weight,
                    boltzmann_temperature=boltzmann_temperature,
                    bmobs_temperature=bmobs_temperature,
                    act_threshold=act_threshold,
                    dspark_floor=dspark_floor,
                    v10_config=v10_config,
                    v11_config=v11_config,
                    v12_model=v12_model,
                    v12_config=v12_config,
                    v14_estimator=v14_estimator,
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
                first_text.setdefault(method, runtime.decode(output))
        examples.append({"prompt": prompt, "reference": runtime.decode(reference), "outputs": first_text})

    summary = {method: _summary(rows[method]) for method in METHODS}
    normal_tps = float(summary["normal"]["tokens_per_second_median"])
    for method in METHODS:
        summary[method]["speedup_vs_normal"] = float(
            summary[method]["tokens_per_second_median"] / max(normal_tps, 1e-12)
        )
    return rows, summary, examples


def _pct_reduction(old: float, new: float) -> float:
    return 100.0 * (1.0 - float(new) / max(float(old), 1e-12))


def _report_html(data: dict) -> str:
    summary = data["summary"]
    speculative = METHODS[1:]
    winner = max(speculative, key=lambda m: summary[m]["tokens_per_second_median"])
    best = summary[winner]
    normal = summary["normal"]
    all_exact = all(summary[m]["all_exact"] for m in METHODS)
    v10 = summary["boltzmann_v10"]
    v12 = summary[V12_METHOD]
    v13 = summary[V13_METHOD]
    v14 = summary[V14_METHOD]
    holdout = data.get("calibration", {}).get(V14_METHOD, {})

    payload = json.dumps(
        {"summary": summary, "methods": METHOD_INFO, "order": METHODS, "winner": winner},
        separators=(",", ":"),
    ).replace("</", "<\\/")

    rows = []
    for method in sorted(METHODS, key=lambda m: summary[m]["speedup_vs_normal"], reverse=True):
        info, s = METHOD_INFO[method], summary[method]
        cls = " winner" if method == winner else (" next" if method in (V13_METHOD, V14_METHOD) else "")
        rows.append(
            f"<tr class='{cls}'><td><span class='no'>{info['index']:02d}</span><b>{html.escape(info['label'])}</b>"
            f"<small>{html.escape(info['short'])}</small></td><td>{s['tokens_per_second_median']:.3f}</td>"
            f"<td><b>{s['speedup_vs_normal']:.3f}×</b></td><td>{100*s['mean_acceptance_rate']:.1f}%</td>"
            f"<td>{s['mean_target_forward_passes']:.2f}</td><td>{s['mean_tokens_per_target_pass']:.3f}</td>"
            f"<td>{s['selection_seconds_median']*1000:.2f} ms</td><td>{s['mean_guidance_work']:.1f}</td>"
            f"<td>{'✓' if s['all_exact'] else '✗'}</td></tr>"
        )

    options = "".join(
        f"<option value='{m}'>{METHOD_INFO[m]['index']:02d} · {html.escape(METHOD_INFO[m]['label'])}</option>"
        for m in speculative
    )
    estimator_text = (
        f"Holdout fine-gap MAE <b>{holdout.get('fine_gap_mae', 0.0):.3f}</b>; "
        f"sign accuracy <b>{100*holdout.get('corrected_sign_accuracy', 0.0):.1f}%</b> "
        f"vs coarse <b>{100*holdout.get('coarse_sign_accuracy', 0.0):.1f}%</b>."
    )

    css = """
:root{font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;background:#06111d;color:#edf7ff;--p:#0d1d2c;--line:#25445d;--muted:#94adbf;--cyan:#5cddff;--green:#82f0b2;--violet:#c2a2ff;--amber:#ffd479}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 8% -5%,#174f75 0,transparent 33%),radial-gradient(circle at 95% 8%,#31205b 0,transparent 27%),#06111d;color:#edf7ff}a{color:var(--cyan)}.wrap{max-width:1340px;margin:auto;padding:26px 22px 80px}.panel{background:linear-gradient(145deg,#10283c,#091724);border:1px solid var(--line);border-radius:22px;box-shadow:0 22px 58px #0004}.top{display:flex;justify-content:space-between;align-items:center;margin-bottom:22px}.brand{font-weight:850}.badge{padding:8px 12px;border:1px solid #355873;border-radius:999px;color:#c9dbe8}.hero{display:grid;grid-template-columns:1.45fr .75fr;gap:16px}.hero>div{padding:32px}.eyebrow{font-size:.76rem;font-weight:850;letter-spacing:.16em;text-transform:uppercase;color:var(--cyan)}h1{font-size:clamp(2.5rem,6vw,5.4rem);line-height:.91;letter-spacing:-.055em;margin:12px 0 18px}h2{font-size:1.75rem;margin:6px 0}p{color:#c0d3e1;line-height:1.6}.stats{display:grid;gap:11px;align-content:center}.stat{padding:16px;border:1px solid #294a63;border-radius:16px;background:#081827}.stat b{display:block;font-size:1.45rem;color:var(--green)}.kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:11px;margin:16px 0 26px}.kpi{padding:17px}.kpi b{font-size:1.45rem;display:block}.kpi span,small{color:var(--muted)}.section{padding:24px;margin-top:17px}.section-head{display:flex;justify-content:space-between;align-items:end;gap:20px}.chart{margin-top:18px}.bar{display:grid;grid-template-columns:245px 1fr 74px;gap:10px;align-items:center;margin:7px 0}.track{height:13px;background:#07131f;border:1px solid #243f54;border-radius:999px;overflow:hidden}.fill{height:100%;background:linear-gradient(90deg,#2d81a5,#5cddff);border-radius:999px}.fill.fast{background:linear-gradient(90deg,#287856,#82f0b2)}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:17px}.compare{padding:23px}.compare h3{font-size:1.45rem;margin:6px 0}.delta{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:15px}.delta div{padding:12px;border-radius:14px;background:#081827;border:1px solid #294a63}.delta b{display:block;font-size:1.15rem}.v13{border-color:#397858}.v14{border-color:#6b55a0}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;margin-top:16px;font-size:.92rem}th,td{padding:12px 10px;border-bottom:1px solid #203b50;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}td:first-child b{display:block}tr.winner{background:#14372555}tr.next{background:#30234b44}.no{display:inline-grid;place-items:center;width:31px;height:31px;margin-right:8px;border-radius:9px;background:#122b3d;color:var(--cyan)}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}.explorer{display:grid;grid-template-columns:.8fr 1.2fr;gap:18px;margin-top:15px}select{width:100%;padding:12px;border-radius:12px;border:1px solid #31516a;background:#081827;color:#edf7ff}.metric-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:9px;margin-top:14px}.metric{padding:12px;border:1px solid #294a63;border-radius:13px;background:#081827}.metric b{display:block}.pipeline{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:18px 0}.stage{padding:14px 8px;text-align:center;border-radius:12px;border:1px solid #294a63;color:var(--muted)}.stage.active{border-color:var(--cyan);color:#fff;box-shadow:0 0 24px #5cddff22}.tokens{display:flex;gap:9px;flex-wrap:wrap}.token{padding:12px 15px;border-radius:12px;background:#17354a;border:1px solid #31546c}.token.accept{background:#17462f;border-color:#3b8b60}.token.reject{background:#512a31;border-color:#9a4b59}.foot{text-align:center;font-size:.86rem;color:var(--muted);margin-top:25px}@media(max-width:900px){.hero,.pair,.grid2,.explorer{grid-template-columns:1fr}.kpis{grid-template-columns:repeat(2,1fr)}.bar{grid-template-columns:145px 1fr 60px}.delta{grid-template-columns:1fr}}
"""
    js = """const D=JSON.parse(document.getElementById('payload').textContent),S=D.summary,M=D.methods,O=D.order;const f=(x,n=3)=>Number(x).toFixed(n);function bars(){const r=document.getElementById('bars'),mx=Math.max(...O.map(k=>S[k].speedup_vs_normal),1.15);r.innerHTML=O.map(k=>`<div class='bar'><div>${String(M[k].index).padStart(2,'0')} · ${M[k].label}</div><div class='track'><div class='fill ${S[k].speedup_vs_normal>=1?'fast':''}' style='width:${100*S[k].speedup_vs_normal/mx}%'></div></div><b>${f(S[k].speedup_vs_normal)}×</b></div>`).join('')}let timer=null;function choose(k){method.value=k;let m=M[k],s=S[k];title.textContent=m.label;desc.textContent=m.mechanism;complexity.textContent=m.complexity;metrics.innerHTML=`<div class='metric'><b>${f(s.tokens_per_second_median)}</b><span>tok/s</span></div><div class='metric'><b>${f(s.speedup_vs_normal)}×</b><span>vs normal</span></div><div class='metric'><b>${(100*s.mean_acceptance_rate).toFixed(1)}%</b><span>acceptance</span></div><div class='metric'><b>${s.mean_guidance_work.toFixed(1)}</b><span>guidance work</span></div>`;let p=0;draw(k,p);if(timer)clearInterval(timer);timer=setInterval(()=>{p=(p+1)%4;draw(k,p)},900)}function draw(k,p){let st=M[k].stages;pipeline.innerHTML=st.map((x,i)=>`<div class='stage ${i==p?'active':''}'>${x}</div>`).join('');let a=p>=2&&S[k].mean_acceptance_rate>.07?1:0;tokens.innerHTML=[0,1,2,3].map(i=>`<div class='token ${p>=2?(i<a?'accept':i==a?'reject':''):''}'>t+${i+1}</div>`).join('')}method.onchange=e=>choose(e.target.value);bars();choose(D.winner);"""

    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><meta name='color-scheme' content='dark'><title>DFlash Mini Lab · LFM2.5 All-14 Study</title><style>{css}</style></head><body><main class='wrap'>
<div class='top'><div class='brand'>DFlash Mini Lab</div><div class='badge'>Canonical LFM2.5 CPU evidence</div></div>
<section class='hero'><div class='panel'><div class='eyebrow'>Normal + fourteen speculative methods</div><h1>LFM2.5<br>All-14 Study</h1><p>One target, one CPU protocol, exact greedy verification. V13 minimizes V10 routing overhead; V14 compresses Parareal into a scalar fine-gap estimator.</p></div><div class='panel stats'><div class='stat'><b>{best['speedup_vs_normal']:.3f}×</b><span>best speedup</span></div><div class='stat'><b>{html.escape(METHOD_INFO[winner]['label'])}</b><span>fastest speculative method</span></div><div class='stat'><b>{'✓ All 15 paths exact' if all_exact else '✗ exactness failure'}</b><span>normal + 14 methods</span></div></div></section>
<section class='kpis'><div class='panel kpi'><b>{normal['tokens_per_second_median']:.3f}</b><span>normal tok/s</span></div><div class='panel kpi'><b>{best['tokens_per_second_median']:.3f}</b><span>best tok/s</span></div><div class='panel kpi'><b>14</b><span>speculative methods</span></div><div class='panel kpi'><b>{data['model']['target_parameter_count']/1e6:.1f}M</b><span>target parameters</span></div><div class='panel kpi'><b>{data['model']['candidate_size']:,}</b><span>draft candidates</span></div></section>
<section class='panel section'><div class='section-head'><div><div class='eyebrow'>Measured performance</div><h2>Speedup versus normal decoding</h2></div><p>Median tokens/s from the same exactness-gated LFM2.5 run.</p></div><div id='bars' class='chart'></div></section>
<section class='pair'><article class='panel compare v13'><div class='eyebrow'>V13 · optimize V10</div><h3>MinOp: keep the policy, remove the plumbing</h3><p>Top-2 is selected inside the Torch drafter; only B×2 IDs/scores cross to NumPy and at most one slot receives the V10 two-way decision.</p><div class='delta'><div><b>{v13['speedup_vs_normal']:.3f}×</b><span>V13 speedup</span></div><div><b>{v13['selection_seconds_median']*1000:.2f} ms</b><span>selection</span></div><div><b>{v13['mean_guidance_work']:.1f}</b><span>guidance</span></div></div><p>V10 reference: {v10['speedup_vs_normal']:.3f}×, {v10['selection_seconds_median']*1000:.2f} ms selection.</p></article>
<article class='panel compare v14'><div class='eyebrow'>V14 · simplify Parareal</div><h3>One scalar F estimator, one residual update</h3><p>Instead of correcting the complete top-k field twice, V14 estimates the target's signed top1-top2 gap with intercept + coarse gap + block position.</p><div class='delta'><div><b>{v14['speedup_vs_normal']:.3f}×</b><span>V14 speedup</span></div><div><b>{_pct_reduction(v12['mean_guidance_work'],v14['mean_guidance_work']):.1f}%</b><span>guidance reduction vs V12</span></div><div><b>{v14['selection_seconds_median']*1000:.2f} ms</b><span>selection</span></div></div><p>{estimator_text}</p></article></section>
<section class='panel section'><div class='section-head'><div><div class='eyebrow'>Exactness-gated evidence</div><h2>Complete benchmark table</h2></div></div><div class='table-wrap'><table><thead><tr><th>Method</th><th>tok/s</th><th>vs normal</th><th>accept</th><th>target calls</th><th>tok/call</th><th>selection</th><th>guidance</th><th>exact</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<section class='panel section'><div class='section-head'><div><div class='eyebrow'>Method explorer</div><h2>Mechanism simulation</h2></div><p>Illustrative mechanism animation; numerical values above are measured.</p></div><div class='explorer'><div><select id='method'>{options}</select><h2 id='title'></h2><p id='desc'></p><p><b>Complexity:</b> <span id='complexity'></span></p><div id='metrics' class='metric-grid'></div></div><div><div id='pipeline' class='pipeline'></div><div id='tokens' class='tokens'></div></div></div></section>
<section class='panel section'><div class='grid2'><div><div class='eyebrow'>Protocol</div><h2>Matched evidence</h2><p><b>Target:</b> {html.escape(data['model']['id'])}</p><p>{data['config']['prompt_count']} held-out prompts × {data['config']['max_new_tokens']} tokens × {data['config']['repeats']} repeats · {data['config']['cpu_threads']} CPU threads · {html.escape(data['config']['dtype'])}.</p></div><div><div class='eyebrow'>Artifacts</div><h2>Reproducible outputs</h2><p>V13 reuses the V10 calibration settings. V14's three coefficients are fit only on training trajectories; benchmark prompts remain held out.</p><p><a href='./benchmark.json'>benchmark.json</a></p></div></div></section>
<p class='foot'>DFlash3–DFlash14 are experimental DFlash Mini Lab variants, not upstream official DFlash releases. Speed claims apply only to the matched LFM2.5 benchmark shown here.</p>
<script id='payload' type='application/json'>{payload}</script><script>{js}</script></main></body></html>"""


def run(args: argparse.Namespace) -> dict:
    runtime = LfmDSparkRuntime(args.aux, args.dspark, cpu_threads=args.cpu_threads, dtype=args.dtype)
    prompts = _read_prompts(args.prompts, args.prompt_limit)
    calibration_prompts = _read_prompts(args.calibration_prompts, args.calibration_prompt_limit)
    if not prompts or not calibration_prompts:
        raise ValueError("benchmark and calibration prompt sets must be non-empty")
    if {p.strip() for p in prompts} & {p.strip() for p in calibration_prompts}:
        raise ValueError("benchmark and calibration prompts must be disjoint")
    if args.tokens < 1 or args.repeats < 1 or args.calibration_tokens < 1:
        raise ValueError("tokens, repeats and calibration tokens must be positive")
    if getattr(args, "tune", False) and getattr(args, "tuning_repeats", 2) < 1:
        raise ValueError("tuning repeats must be positive")

    warm_ids = runtime.encode(prompts[0])
    _ = runtime.target_logits(warm_ids)
    _ = runtime.draft_logits(runtime.context_features(warm_ids))

    print("Calibrating ACT, DSpark, V10 and V11 on separate prompts...", flush=True)
    act_threshold, act_cal = calibrate_act(runtime, calibration_prompts, tokens=args.calibration_tokens)
    dspark_floor, dspark_cal = calibrate_dspark(
        runtime, calibration_prompts, tokens=args.calibration_tokens, top_k=args.top_k
    )
    v10_config, v10_cal = calibrate_v10(runtime, calibration_prompts, tokens=args.calibration_tokens)
    v11_config, v11_cal = calibrate_v11(runtime, calibration_prompts, tokens=args.calibration_tokens)
    v12_model = load_linear_model(args.v12_model)
    v12_config = base._v12_config(v12_model, args.top_k)
    v14_estimator = load_estimator(args.v14_model)

    width_calibration = {"enabled": False}
    if getattr(args, "tune", False):
        print("Calibrating verification widths for all 14 methods...", flush=True)
        def trial(method, ids, tokens):
            return _run_method(
                method, runtime, ids, tokens,
                top_k=args.top_k, jump_weight=args.jump_weight,
                fused_weight=args.fused_weight,
                boltzmann_temperature=args.boltzmann_temperature,
                bmobs_temperature=args.bmobs_temperature,
                act_threshold=act_threshold, dspark_floor=dspark_floor,
                v10_config=v10_config, v11_config=v11_config,
                v12_model=v12_model, v12_config=v12_config,
                v14_estimator=v14_estimator,
            )
        runtime.verify_widths, width_calibration = calibrate_verify_widths(
            runtime, calibration_prompts, METHODS[1:], trial,
            tokens=args.calibration_tokens,
            repeats=getattr(args, "tuning_repeats", 2),
        )
        width_calibration["enabled"] = True

    print("Benchmarking Normal + all 14 methods on held-out prompts...", flush=True)
    rows, summary, examples = benchmark(
        runtime,
        prompts,
        tokens=args.tokens,
        repeats=args.repeats,
        top_k=args.top_k,
        jump_weight=args.jump_weight,
        fused_weight=args.fused_weight,
        boltzmann_temperature=args.boltzmann_temperature,
        bmobs_temperature=args.bmobs_temperature,
        act_threshold=act_threshold,
        dspark_floor=dspark_floor,
        v10_config=v10_config,
        v11_config=v11_config,
        v12_model=v12_model,
        v12_config=v12_config,
        v14_estimator=v14_estimator,
    )
    winner = max(METHODS[1:], key=lambda m: summary[m]["tokens_per_second_median"])
    data = {
        "study": "LFM2.5 All-14 speculative decoding study",
        "model": {
            "id": runtime.model_id,
            "target_parameter_count": int(runtime.target_parameter_count),
            "target_vocab_size": int(runtime.target.config.vocab_size),
            "candidate_size": int(runtime.candidate_size),
            "aux_parameter_count": int(runtime.aux_parameter_count),
        },
        "config": {
            "max_new_tokens": int(args.tokens),
            "repeats": int(args.repeats),
            "prompt_count": len(prompts),
            "calibration_prompt_count": len(calibration_prompts),
            "calibration_tokens": int(args.calibration_tokens),
            "top_k": int(args.top_k),
            "jump_weight": float(args.jump_weight),
            "fused_weight": float(args.fused_weight),
            "boltzmann_temperature": float(args.boltzmann_temperature),
            "bmobs_temperature": float(args.bmobs_temperature),
            "cpu_threads": int(args.cpu_threads),
            "dtype": args.dtype,
            "target_suffix_projection": runtime.optimize_target_logits and runtime._supports_logits_to_keep,
            "target_argmax_on_device": runtime.optimize_target_logits,
            "bonus_token": runtime.use_bonus_token,
            "target_cache": False,
            "verify_width_tuning": bool(getattr(args, "tune", False)),
        },
        "selected_settings": {
            "verify_widths": getattr(runtime, "verify_widths", {}),
            "act_margin_threshold": float(act_threshold),
            "dspark_survival_floor": float(dspark_floor),
            "v10": getattr(v10_config, "__dict__", {}),
            "v13": {
                "policy_source": "V10 selected config",
                "routing_budget": 1,
                "top2_location": "Torch drafter before NumPy transfer",
            },
            "v11": getattr(v11_config, "__dict__", {}),
            "v12": {
                "top_k": v12_config.top_k,
                "correction_rounds": v12_config.correction_rounds,
                "damping": v12_config.damping,
            },
            "v14": {
                "damping": v14_estimator.damping,
                "switch_threshold": v14_estimator.switch_threshold,
                "max_corrections": v14_estimator.max_corrections,
                "coefficients": {
                    "intercept": v14_estimator.intercept,
                    "coarse_gap_weight": v14_estimator.coarse_gap_weight,
                    "position_weight": v14_estimator.position_weight,
                },
            },
        },
        "calibration": {
            "verify_widths": width_calibration,
            "dflash7_act": act_cal,
            "dspark_v9": dspark_cal,
            "boltzmann_v10": v10_cal,
            V11_METHOD: v11_cal,
            V12_METHOD: (v12_model.metadata or {}).get("holdout_convergence", {}),
            V13_METHOD: {"reuses_v10_config": True, "routing_budget": 1},
            V14_METHOD: (v14_estimator.metadata or {}).get("holdout_metrics", {}),
        },
        "method_info": METHOD_INFO,
        "summary": summary,
        "rows": rows,
        "examples": examples,
        "winner": winner,
    }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = _report_html(data)
    (out / "report.html").write_text(report, encoding="utf-8")
    (out / "index.html").write_text(report, encoding="utf-8")
    if not all(item["all_exact"] for item in summary.values()):
        raise RuntimeError(f"Exactness check failed; diagnostic results saved to {out / 'benchmark.json'}")
    return data


def main() -> None:
    p = argparse.ArgumentParser(description="Run the canonical Normal + 14-method LFM2.5 CPU benchmark")
    p.add_argument("--aux", default="lfm-artifacts/lfm_aux.pt")
    p.add_argument("--dspark", default="lfm-artifacts/lfm_dspark.pt")
    p.add_argument("--v12-model", default="lfm-artifacts/v12_parareal.json")
    p.add_argument("--v14-model", default="lfm-artifacts/v14_simple_parareal.json")
    p.add_argument("--prompts", default="real_benchmarks/prompts.json")
    p.add_argument("--calibration-prompts", default="real_benchmarks/calibration_prompts.json")
    p.add_argument("--output-dir", default="lfm-reports")
    p.add_argument("--tokens", type=int, default=24)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--prompt-limit", type=int, default=6)
    p.add_argument("--calibration-tokens", type=int, default=8)
    p.add_argument("--calibration-prompt-limit", type=int, default=4)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--jump-weight", type=float, default=0.5)
    p.add_argument("--fused-weight", type=float, default=1.0)
    p.add_argument("--boltzmann-temperature", type=float, default=0.15)
    p.add_argument("--bmobs-temperature", type=float, default=0.35)
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--tune", action="store_true", help="Calibrate verification width per method on separate prompts")
    p.add_argument("--tuning-repeats", type=int, default=2)
    p.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    args = p.parse_args()
    data = run(args)
    print(json.dumps({"winner": data["winner"], "model": data["model"]["id"], "summary": data["summary"]}, indent=2))


if __name__ == "__main__":
    main()
