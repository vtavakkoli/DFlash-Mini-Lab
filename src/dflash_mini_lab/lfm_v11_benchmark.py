from __future__ import annotations

import argparse
import html
import json
import statistics
import time
from pathlib import Path

import numpy as np

from .lfm_benchmark import RealDecodeStats, _read_prompts, normal_decode, speculative_decode
from .lfm_dspark import LfmDSparkRuntime
from .lfm_v9v10_benchmark import (
    _greedy_reference,
    dspark_decode,
    run as run_v10_study,
    v10_decode,
)
from .lfm_v10 import V10Config
from .lfm_v11 import V11Config, config_dict, config_key, select_v11_boltzmann_gated_mobs, v11_grid


V11_METHOD = "boltzmann_gated_mobs_v11"
FOCUS_METHODS = (
    "normal",
    "dflash",
    "dflash3_mobs",
    "dspark_v9",
    "boltzmann_v10",
    V11_METHOD,
)
LABELS = {
    "normal": "Normal LFM",
    "dflash": "DFlash",
    "dflash3_mobs": "DFlash3-MOBS",
    "dspark_v9": "V9 DSpark-Lite",
    "boltzmann_v10": "V10 Advanced Boltzmann",
    V11_METHOD: "V11 Boltzmann-Gated MOBS",
}


def _verify(runtime: LfmDSparkRuntime, seq: np.ndarray, proposal: np.ndarray):
    verify_input = np.concatenate([seq, proposal])
    t = time.perf_counter()
    logits = runtime.target_logits(verify_input)
    elapsed = time.perf_counter() - t
    p, k = int(seq.size), int(proposal.size)
    verifier = np.argmax(logits[p - 1 : p - 1 + k], axis=-1).astype(np.int64)
    mismatch = np.flatnonzero(proposal != verifier)
    accepted = k if mismatch.size == 0 else int(mismatch[0])
    return verifier, accepted, elapsed


def v11_decode(
    runtime: LfmDSparkRuntime,
    input_ids: np.ndarray,
    max_new_tokens: int,
    *,
    config: V11Config,
):
    seq = np.asarray(input_ids, dtype=np.int64).copy()
    start_len = int(seq.size)
    target_calls = draft_calls = accepted_total = proposed_total = 0
    target_seconds = context_seconds = draft_seconds = selection_seconds = 0.0
    pair_scores = gate_evals = gated = eligible = fast = 0
    uncertainty_sum = 0.0
    wall0 = time.perf_counter()

    while int(seq.size) - start_len < int(max_new_tokens):
        remaining = int(max_new_tokens) - (int(seq.size) - start_len)
        t = time.perf_counter()
        context = runtime.context_features(seq)
        context_seconds += time.perf_counter() - t

        t = time.perf_counter()
        draft_logits = runtime.draft_logits(context)
        draft_seconds += time.perf_counter() - t
        draft_calls += 1

        t = time.perf_counter()
        full, meta = select_v11_boltzmann_gated_mobs(
            runtime,
            draft_logits,
            context,
            int(seq[-1]),
            config,
        )
        proposal = np.asarray(full[: min(int(full.size), remaining)], dtype=np.int64)
        block_evals = min(int(full.size), remaining)
        gate_evals += block_evals
        pair_scores += int(meta["pair_scores"])
        gated += int(meta["gated_positions"])
        eligible += int(meta["eligible_positions"])
        fast += int(meta["fast_argmax_positions"])
        uncertainty_sum += float(meta.get("mean_uncertainty", 0.0)) * block_evals
        selection_seconds += time.perf_counter() - t
        proposed_total += int(proposal.size)

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
        method=V11_METHOD,
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
        selector_pair_scores=pair_scores,
        boltzmann_candidate_scores=gate_evals,
    )
    meta = {
        "v11_pair_scores": int(pair_scores),
        "v11_gate_evaluations": int(gate_evals),
        "v11_gated_positions": int(gated),
        "v11_eligible_positions": int(eligible),
        "v11_fast_argmax_positions": int(fast),
        "v11_mean_uncertainty": float(uncertainty_sum / max(gate_evals, 1)),
    }
    return seq, stats, meta


def _aggregate(rows: list[dict], tokens: int) -> dict:
    wall = sum(float(x["wall_seconds"]) for x in rows)
    total = int(tokens) * len(rows)
    return {
        "tokens_per_second": float(total / max(wall, 1e-12)),
        "mean_acceptance_rate": float(statistics.fmean(float(x["acceptance_rate"]) for x in rows)),
        "mean_target_forward_passes": float(statistics.fmean(float(x["target_forward_passes"]) for x in rows)),
        "mean_guidance_scores": float(statistics.fmean(float(x["total_guidance_scores"]) for x in rows)),
        "all_exact": bool(all(bool(x["exact_match"]) for x in rows)),
    }


def _eval_v11_configs(
    runtime: LfmDSparkRuntime,
    prompts: list[str],
    references: dict[str, np.ndarray],
    configs: tuple[V11Config, ...] | list[V11Config],
    tokens: int,
):
    results = []
    for cfg in configs:
        rows = []
        for prompt in prompts:
            ids = runtime.encode(prompt)
            out, stats, meta = v11_decode(runtime, ids, tokens, config=cfg)
            row = stats.to_dict()
            row.update(meta)
            row["exact_match"] = bool(np.array_equal(out, references[prompt]))
            rows.append(row)
        agg = _aggregate(rows, tokens)
        agg["config"] = config_dict(cfg)
        agg["config_key"] = config_key(cfg)
        agg["mean_pair_scores"] = float(statistics.fmean(float(x["v11_pair_scores"]) for x in rows))
        agg["mean_gated_positions"] = float(statistics.fmean(float(x["v11_gated_positions"]) for x in rows))
        results.append(agg)
    return results


def calibrate_v11(runtime: LfmDSparkRuntime, prompts: list[str], *, tokens: int):
    grid = v11_grid()
    references = {p: _greedy_reference(runtime, runtime.encode(p), tokens) for p in prompts}

    # Successive halving: one-prompt screening then full held-out calibration.
    stage1 = _eval_v11_configs(runtime, prompts[:1], references, grid, tokens)
    valid1 = [x for x in stage1 if x["all_exact"]]
    valid1.sort(key=lambda x: (x["tokens_per_second"], -x["mean_guidance_scores"]), reverse=True)
    survivor_keys = {x["config_key"] for x in valid1[:4]}
    survivors = [cfg for cfg in grid if config_key(cfg) in survivor_keys]
    stage2 = _eval_v11_configs(runtime, prompts, references, survivors, tokens)
    valid2 = [x for x in stage2 if x["all_exact"]]
    if not valid2:
        raise RuntimeError("No exact V11 calibration setting")
    best = max(valid2, key=lambda x: (x["tokens_per_second"], -x["mean_guidance_scores"]))
    selected = next(cfg for cfg in survivors if config_key(cfg) == best["config_key"])
    full = len(grid) * len(prompts)
    used = len(grid) + len(survivors) * len(prompts)
    return selected, {
        "stage1": stage1,
        "stage2": stage2,
        "selected": best,
        "plan": {
            "total_configs": len(grid),
            "stage1_prompts": 1,
            "survivors": len(survivors),
            "full_grid_prompt_config_evaluations": full,
            "successive_halving_prompt_config_evaluations": used,
            "calibration_work_reduction_fraction": 1.0 - used / max(full, 1),
            "training_steps": 0,
        },
    }


def _summary(rows: list[dict]) -> dict:
    def mean(key: str) -> float:
        return float(statistics.fmean(float(x[key]) for x in rows))

    def median(key: str) -> float:
        return float(statistics.median(float(x[key]) for x in rows))

    return {
        "tokens_per_second_median": median("tokens_per_second"),
        "latency_seconds_median": median("wall_seconds"),
        "target_seconds_median": median("target_seconds"),
        "draft_seconds_median": median("draft_seconds"),
        "selection_seconds_median": median("selection_seconds"),
        "mean_target_forward_passes": mean("target_forward_passes"),
        "mean_draft_forward_passes": mean("draft_forward_passes"),
        "mean_acceptance_rate": mean("acceptance_rate"),
        "mean_tokens_per_target_pass": mean("tokens_per_target_pass"),
        "mean_total_guidance_scores": mean("total_guidance_scores"),
        "all_exact": bool(all(bool(x["exact_match"]) for x in rows)),
    }


def benchmark_focus(
    runtime: LfmDSparkRuntime,
    prompts: list[str],
    *,
    tokens: int,
    repeats: int,
    top_k: int,
    dspark_floor: float,
    v10_config: V10Config,
    v11_config: V11Config,
):
    rows = {m: [] for m in FOCUS_METHODS}
    outputs = []
    for pi, prompt in enumerate(prompts):
        ids = runtime.encode(prompt)
        reference = _greedy_reference(runtime, ids, tokens)
        first: dict[str, str] = {}
        for repeat in range(max(1, int(repeats))):
            shift = (pi * max(1, int(repeats)) + repeat) % len(FOCUS_METHODS)
            order = FOCUS_METHODS[shift:] + FOCUS_METHODS[:shift]
            for method in order:
                if method == "normal":
                    out, stats = normal_decode(runtime, ids, tokens)
                    meta = {}
                elif method == "dflash":
                    out, stats = speculative_decode(runtime, ids, tokens, "dflash", top_k=top_k)
                    meta = {}
                elif method == "dflash3_mobs":
                    out, stats = speculative_decode(runtime, ids, tokens, "dflash3_mobs", top_k=top_k)
                    meta = {}
                elif method == "dspark_v9":
                    out, stats, meta = dspark_decode(
                        runtime,
                        ids,
                        tokens,
                        top_k=top_k,
                        survival_floor=dspark_floor,
                    )
                elif method == "boltzmann_v10":
                    out, stats, meta = v10_decode(runtime, ids, tokens, config=v10_config)
                else:
                    out, stats, meta = v11_decode(runtime, ids, tokens, config=v11_config)
                exact = bool(np.array_equal(out, reference))
                row = stats.to_dict()
                row.update(meta)
                row.update(prompt=prompt, prompt_index=pi, repeat=repeat, exact_match=exact)
                rows[method].append(row)
                first.setdefault(method, runtime.decode(out))
        outputs.append({"prompt": prompt, "reference": runtime.decode(reference), "texts": first})
    summary = {m: _summary(rows[m]) for m in FOCUS_METHODS}
    normal_tps = float(summary["normal"]["tokens_per_second_median"])
    for method in FOCUS_METHODS:
        summary[method]["speedup_vs_normal"] = float(
            summary[method]["tokens_per_second_median"] / max(normal_tps, 1e-12)
        )
    summary[V11_METHOD]["mean_pair_scores"] = float(
        statistics.fmean(float(x["v11_pair_scores"]) for x in rows[V11_METHOD])
    )
    summary[V11_METHOD]["mean_gate_evaluations"] = float(
        statistics.fmean(float(x["v11_gate_evaluations"]) for x in rows[V11_METHOD])
    )
    summary[V11_METHOD]["mean_gated_positions"] = float(
        statistics.fmean(float(x["v11_gated_positions"]) for x in rows[V11_METHOD])
    )
    summary[V11_METHOD]["mean_fast_argmax_positions"] = float(
        statistics.fmean(float(x["v11_fast_argmax_positions"]) for x in rows[V11_METHOD])
    )
    return rows, summary, outputs


def build_report(data: dict, path: str | Path) -> None:
    focus = data["v11"]["focus_summary"]
    winner = data["v11"]["focus_winner"]
    sorted_methods = sorted(FOCUS_METHODS, key=lambda m: focus[m]["tokens_per_second_median"], reverse=True)
    rows = "".join(
        f"<tr class='{'win' if m == winner else ''}'><th>{html.escape(LABELS[m])}</th>"
        f"<td>{focus[m]['tokens_per_second_median']:.3f}</td>"
        f"<td>{focus[m]['speedup_vs_normal']:.3f}×</td>"
        f"<td>{100*focus[m]['mean_acceptance_rate']:.1f}%</td>"
        f"<td>{focus[m]['mean_tokens_per_target_pass']:.2f}</td>"
        f"<td>{focus[m]['selection_seconds_median']:.4f}s</td>"
        f"<td>{focus[m]['mean_total_guidance_scores']:.1f}</td>"
        f"<td>{'✓' if focus[m]['all_exact'] else '✗'}</td></tr>"
        for m in sorted_methods
    )
    cfg = data["v11"]["selected_config"]
    stage2 = data["v11"]["calibration"]["stage2"]
    stage_rows = "".join(
        f"<tr><td>{html.escape(x['config_key'])}</td><td>{x['tokens_per_second']:.3f}</td>"
        f"<td>{100*x['mean_acceptance_rate']:.1f}%</td><td>{x['mean_guidance_scores']:.1f}</td></tr>"
        for x in sorted(stage2, key=lambda z: z["tokens_per_second"], reverse=True)
    )
    js = json.dumps(
        {m: {"label": LABELS[m], "tps": focus[m]["tokens_per_second_median"], "accept": focus[m]["mean_acceptance_rate"]} for m in FOCUS_METHODS},
        separators=(",", ":"),
    )
    v = focus[V11_METHOD]
    mobs_ops = float(focus["dflash3_mobs"]["mean_total_guidance_scores"])
    v11_ops = float(v["mean_total_guidance_scores"])
    op_reduction = 1.0 - v11_ops / max(mobs_ops, 1e-9)
    lead = float(data["v11"]["lead_vs_next_fraction"])
    css = """
    :root{font-family:Inter,ui-sans-serif,system-ui;background:#07111f;color:#eef7ff;--p:#0d1c2e;--line:#284761;--cyan:#58d9ff;--green:#7cf5b2;--muted:#98abc0}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% 0,#173b5a,transparent 35%),#07111f}.wrap{max-width:1220px;margin:auto;padding:28px 20px 70px}.hero,.card,.demo{background:linear-gradient(145deg,#10253b,#0a1727);border:1px solid var(--line);border-radius:22px;padding:24px;margin-bottom:20px}h1{font-size:clamp(2.2rem,5vw,4.5rem);line-height:.96;margin:.25em 0}.chips{display:flex;gap:8px;flex-wrap:wrap}.chip{border:1px solid #345977;border-radius:999px;padding:6px 10px;color:#bcecff}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.metric{border:1px solid var(--line);border-radius:15px;padding:14px;background:#091727}.metric b{display:block;font-size:1.55rem;color:var(--green)}.muted{color:var(--muted)}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.table{overflow:auto}table{width:100%;border-collapse:collapse;min-width:850px}th,td{padding:10px;border-bottom:1px solid #20384f;text-align:right}th:first-child{text-align:left}.win{background:#123b2a}.flow{display:grid;grid-template-columns:repeat(6,1fr);gap:9px}.node{min-height:82px;border:1px solid #2d4e69;border-radius:14px;display:grid;place-items:center;text-align:center;padding:8px}.node.on{border-color:var(--cyan);box-shadow:0 0 25px #58d9ff44}.race{display:grid;grid-template-columns:190px 1fr 50px;gap:9px;align-items:center;margin:10px 0}.track{height:18px;border:1px solid #29475f;border-radius:999px;overflow:hidden}.fill{height:100%;width:0;background:linear-gradient(90deg,var(--cyan),var(--green))}select,button{background:#10263b;color:#eef7ff;border:1px solid #365b78;border-radius:10px;padding:9px}.slots{display:flex;gap:8px;flex-wrap:wrap}.slot{padding:12px;border:1px solid #365773;border-radius:11px;background:#10243a}.slot.fast{background:#12362b;border-color:#49c78d}.slot.mobs{background:#382d12;border-color:#d5a640}@media(max-width:850px){.grid{grid-template-columns:1fr}.metrics{grid-template-columns:1fr 1fr}.flow{grid-template-columns:1fr 1fr}}
    """
    doc = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>V11 Boltzmann-Gated MOBS</title><style>{css}</style></head><body><main class='wrap'>
    <section class='hero'><div class='chips'><span class='chip'>LFM2.5-350M-Base</span><span class='chip'>float32 CPU · 2 threads</span><span class='chip'>rotated execution order</span><span class='chip'>exact greedy output</span></div><h1>V11 · Boltzmann-Gated MOBS</h1><p class='muted'>V10's cheap two-way Boltzmann probability is used only as an uncertainty router. Confident slots stay DFlash argmax; a tiny budget of uncertain slots pays MOBS dependency scoring.</p><div class='metrics'><div class='metric'><small>Current contender winner</small><b>{html.escape(LABELS[winner])}</b></div><div class='metric'><small>Winner speed</small><b>{focus[winner]['tokens_per_second_median']:.3f} tok/s</b></div><div class='metric'><small>Lead vs #2</small><b>{100*lead:.2f}%</b></div><div class='metric'><small>V11 vs full MOBS guidance</small><b>{100*op_reduction:.1f}% less</b></div></div></section>
    <section class='card'><h2>Authoritative focused race</h2><p class='muted'>Six current contenders are interleaved with a rotating execution order across 6 held-out prompts × 24 generated tokens × 3 repeats. This reduces method-order and CPU thermal bias.</p><div class='table'><table><thead><tr><th>Method</th><th>tok/s</th><th>vs Normal</th><th>Acceptance</th><th>Tokens/pass</th><th>Select time</th><th>Guidance ops</th><th>Exact</th></tr></thead><tbody>{rows}</tbody></table></div></section>
    <section class='grid'><div class='card'><h2>Selected V11 configuration</h2><p><code>{html.escape(config_key(V11Config(**cfg)))}</code></p><p><b>Boltzmann temperature:</b> {cfg['temperature']}<br><b>Gate floor:</b> {cfg['gate_floor']}<br><b>MOBS budget:</b> {cfg['mobs_budget']} / 4 slots<br><b>MOBS top-k:</b> {cfg['top_k']}</p><p class='muted'>No V11 training. Calibration is held out from the benchmark and uses successive halving.</p></div><div class='card'><h2>Measured V11 work</h2><p><b>{v['mean_gated_positions']:.1f}</b> MOBS-gated positions/run · <b>{v['mean_fast_argmax_positions']:.1f}</b> fast argmax positions/run.</p><p><b>{v['mean_pair_scores']:.1f}</b> pair scores + <b>{v['mean_gate_evaluations']:.1f}</b> Boltzmann gate evaluations per run.</p><p class='muted'>The target verifier is still authoritative; V11 cannot change the greedy output.</p></div></section>
    <section class='card'><h2>Animated architecture</h2><div class='flow'><div class='node'>Prompt/context</div><div class='node'>Parallel DFlash draft</div><div class='node'>Top-2 Boltzmann uncertainty</div><div class='node'>Gate only weak slots</div><div class='node'>Sparse MOBS correction</div><div class='node'>Exact LFM verification</div></div></section>
    <section class='grid'><div class='demo'><h2>Live demo 1 · measured race</h2><p><select id='a'></select> <select id='b'></select> <button id='go'>Race 48 tokens</button></p><div class='race'><b id='al'></b><div class='track'><div id='af' class='fill'></div></div><span id='ao'>0</span></div><div class='race'><b id='bl'></b><div class='track'><div id='bf' class='fill'></div></div><span id='bo'>0</span></div><p id='verdict' class='muted'>Ratios come from measured throughput; browser time is accelerated.</p></div><div class='demo'><h2>Live demo 2 · V11 gate</h2><p><button id='gate'>Animate one four-slot block</button></p><div id='slots' class='slots'></div><p id='gateNote' class='muted'>Green = plain DFlash fast path; amber = MOBS correction.</p></div></section>
    <section class='card'><h2>Held-out V11 calibration finalists</h2><div class='table'><table><thead><tr><th>Configuration</th><th>tok/s</th><th>Acceptance</th><th>Guidance</th></tr></thead><tbody>{stage_rows}</tbody></table></div></section>
    <section class='card'><h2>Methodology boundary</h2><p>V11 is a mechanism-level experiment in DFlash Mini Lab, not an upstream DFlash/DSpark checkpoint. Its success criterion is end-to-end CPU throughput after all routing and MOBS overhead, with exact token-for-token target verification.</p></section>
    </main><script>const D={js};const methods=Object.keys(D),$=id=>document.getElementById(id);function opts(id,sel){{$ (id).innerHTML=methods.map(m=>`<option value="${{m}}" ${{m===sel?'selected':''}}>${{D[m].label}}</option>`).join('')}}opts('a','dflash3_mobs');opts('b','{V11_METHOD}');let timer=null;$('go').onclick=()=>{{clearInterval(timer);const a=$('a').value,b=$('b').value;$('al').textContent=D[a].label;$('bl').textContent=D[b].label;const total=48,start=performance.now(),scale=.16/Math.max(D[a].tps,D[b].tps);timer=setInterval(()=>{{const e=(performance.now()-start)/1000,ta=Math.min(total,e*D[a].tps/scale),tb=Math.min(total,e*D[b].tps/scale);$('af').style.width=(100*ta/total)+'%';$('bf').style.width=(100*tb/total)+'%';$('ao').textContent=Math.floor(ta);$('bo').textContent=Math.floor(tb);if(ta>=total||tb>=total){{clearInterval(timer);const w=ta>=total?a:b;$('verdict').textContent=D[w].label+' reaches 48 tokens first in the measured ratio.'}}}},40)}};$('gate').onclick=()=>{{$('slots').innerHTML='';const budget={int(cfg['mobs_budget'])},count=Math.min(4,budget),marks=[];for(let i=0;i<4;i++)marks.push(i<count?'mobs':'fast');marks.sort(()=>0.5-Math.random());marks.forEach((kind,i)=>{{const e=document.createElement('span');e.className='slot';e.textContent='slot '+(i+1);$('slots').appendChild(e);setTimeout(()=>{{e.classList.add(kind);e.textContent+=(kind==='mobs'?' · MOBS':' · fast')}},250*i)}});document.querySelectorAll('.node').forEach((n,i)=>setTimeout(()=>{{n.classList.add('on');setTimeout(()=>n.classList.remove('on'),330)}},i*260))}};</script></body></html>"""
    Path(path).write_text(doc, encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    # Reproduce V9/V10 and all established LFM methods first.
    base = run_v10_study(args)
    runtime = LfmDSparkRuntime(
        args.aux,
        args.dspark,
        model_id=args.model_id,
        cpu_threads=args.cpu_threads,
        dtype=args.dtype,
    )
    benchmark_prompts = _read_prompts(args.prompts, args.prompt_limit)
    calibration_prompts = _read_prompts(args.calibration_prompts, args.calibration_prompt_limit)
    warm = runtime.encode(calibration_prompts[0])
    _ = runtime.target_logits(warm)
    _ = runtime.draft_logits(runtime.context_features(warm))

    selected, calibration = calibrate_v11(runtime, calibration_prompts, tokens=args.calibration_tokens)
    dspark_floor = float(base["dspark_v9"]["selected_survival_floor"])
    v10_config = V10Config(**base["boltzmann_v10"]["selected_config"])
    rows, focus_summary, outputs = benchmark_focus(
        runtime,
        benchmark_prompts,
        tokens=args.tokens,
        repeats=args.repeats,
        top_k=args.top_k,
        dspark_floor=dspark_floor,
        v10_config=v10_config,
        v11_config=selected,
    )
    focus_winner = max(FOCUS_METHODS, key=lambda m: focus_summary[m]["tokens_per_second_median"])
    ordered = sorted(FOCUS_METHODS, key=lambda m: focus_summary[m]["tokens_per_second_median"], reverse=True)
    best_tps = float(focus_summary[ordered[0]]["tokens_per_second_median"])
    second_tps = float(focus_summary[ordered[1]]["tokens_per_second_median"])
    lead = best_tps / max(second_tps, 1e-12) - 1.0

    # Primary summaries for the current contenders come from the rotated race.
    for method in FOCUS_METHODS:
        base["summary"][method] = focus_summary[method]
    base["summary"][V11_METHOD] = focus_summary[V11_METHOD]
    base["runs"].extend(rows[V11_METHOD])
    base["winner"] = focus_winner
    base["showcase"]["version"] = 11
    base["method_scope"] = (
        "V11 uses a deterministic two-way Boltzmann uncertainty probability to route only weak DFlash slots "
        "through sparse MOBS pairwise correction. The LFM target verifier remains exact and authoritative. "
        "The current-contender ranking uses rotated execution order to reduce CPU method-order bias."
    )
    base["v11"] = {
        "selected_config": config_dict(selected),
        "calibration": calibration,
        "training_steps": 0,
        "focus_methods": list(FOCUS_METHODS),
        "focus_summary": focus_summary,
        "focus_winner": focus_winner,
        "lead_vs_next_fraction": float(lead),
        "clear_winner_threshold_fraction": 0.02,
        "clear_winner": bool(focus_winner == V11_METHOD and lead >= 0.02),
        "outputs": outputs,
        "mechanism": "DFlash argmax fast path + deterministic Boltzmann uncertainty gate + sparse MOBS correction only on budgeted weak slots.",
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(base, indent=2), encoding="utf-8")
    build_report(base, out / "report.html")
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description="LFM2.5 V11 Boltzmann-Gated MOBS study")
    parser.add_argument("--aux", default="lfm-artifacts/lfm_aux.pt")
    parser.add_argument("--dspark", default="lfm-artifacts/lfm_dspark.pt")
    parser.add_argument("--prompts", default="real_benchmarks/prompts.json")
    parser.add_argument("--calibration-prompts", default="real_benchmarks/calibration_prompts.json")
    parser.add_argument("--output-dir", default="lfm-reports")
    parser.add_argument("--model-id", default="LiquidAI/LFM2.5-350M-Base")
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prompt-limit", type=int, default=6)
    parser.add_argument("--calibration-tokens", type=int, default=8)
    parser.add_argument("--calibration-prompt-limit", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--jump-weight", type=float, default=0.5)
    parser.add_argument("--fused-weight", type=float, default=1.0)
    parser.add_argument("--fused-min-margin", type=float, default=0.0)
    parser.add_argument("--boltzmann-temperature", type=float, default=0.15)
    parser.add_argument("--bmobs-temperature", type=float, default=0.35)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--dtype", choices=("float32",), default="float32")
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps({
        "winner": payload["winner"],
        "focus_summary": payload["v11"]["focus_summary"],
        "v11": {
            "selected_config": payload["v11"]["selected_config"],
            "lead_vs_next_fraction": payload["v11"]["lead_vs_next_fraction"],
            "clear_winner": payload["v11"]["clear_winner"],
        },
    }, indent=2))


if __name__ == "__main__":
    main()
