from __future__ import annotations

import argparse
import html
import json
import statistics
import time
from pathlib import Path

import numpy as np

from .lfm_benchmark import RealDecodeStats, _read_prompts
from .lfm_dspark import LfmDSparkRuntime, confidence_verify_length
from .lfm_showcase import SHOWCASE_LABELS, SHOWCASE_METHODS, run as run_existing_showcase
from .lfm_v10 import V10Config, config_dict, config_key, select_v10_quickpath, successive_halving_plan, v10_grid


EXTRA_LABELS = {
    "dspark_v9": "V9 DSpark-Lite",
    "boltzmann_v10": "V10 Advanced Boltzmann",
}
ALL_LABELS = {**SHOWCASE_LABELS, **EXTRA_LABELS}
ALL_METHODS = tuple(SHOWCASE_METHODS) + ("dspark_v9", "boltzmann_v10")
DSPARK_FLOORS = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70)


def _greedy_reference(runtime: LfmDSparkRuntime, ids: np.ndarray, tokens: int) -> np.ndarray:
    seq = np.asarray(ids, dtype=np.int64).copy()
    for _ in range(int(tokens)):
        logits = runtime.target_logits(seq)
        seq = np.append(seq, int(np.argmax(logits[-1])))
    return seq


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


def dspark_decode(runtime: LfmDSparkRuntime, input_ids: np.ndarray, max_new_tokens: int, *, top_k: int, survival_floor: float, markov_weight: float = 1.0):
    seq = np.asarray(input_ids, dtype=np.int64).copy()
    start_len = int(seq.size)
    target_calls = draft_calls = accepted_total = proposed_total = 0
    target_seconds = context_seconds = draft_seconds = selection_seconds = 0.0
    markov_scores = confidence_evals = trimmed = 0
    confidence_sum = 0.0
    wall0 = time.perf_counter()
    while int(seq.size) - start_len < int(max_new_tokens):
        remaining = int(max_new_tokens) - (int(seq.size) - start_len)
        t = time.perf_counter(); context = runtime.context_features(seq); context_seconds += time.perf_counter() - t
        t = time.perf_counter(); hidden, draft_logits = runtime.draft_hidden_and_logits(context); draft_seconds += time.perf_counter() - t
        draft_calls += 1
        t = time.perf_counter()
        full, confidences, score_ops = runtime.dspark_select_path(hidden, draft_logits, int(seq[-1]), top_k=int(top_k), markov_weight=float(markov_weight))
        raw_len = min(int(full.size), remaining)
        keep = confidence_verify_length(confidences, float(survival_floor), raw_len)
        keep = min(max(1, int(keep)), raw_len)
        proposal = np.asarray(full[:keep], dtype=np.int64)
        markov_scores += int(score_ops); confidence_evals += raw_len
        confidence_sum += float(np.asarray(confidences[:raw_len], dtype=np.float64).sum())
        trimmed += raw_len - keep
        selection_seconds += time.perf_counter() - t
        proposed_total += int(proposal.size)
        verifier, accepted, elapsed = _verify(runtime, seq, proposal)
        target_seconds += elapsed; target_calls += 1; accepted_total += int(accepted)
        if accepted: seq = np.concatenate([seq, proposal[:accepted]])
        if accepted < int(proposal.size) and int(seq.size) - start_len < int(max_new_tokens): seq = np.append(seq, verifier[accepted])
    seq = seq[: start_len + int(max_new_tokens)]
    wall = time.perf_counter() - wall0
    stats = RealDecodeStats(method="dspark_v9", new_tokens=int(max_new_tokens), target_forward_passes=target_calls, draft_forward_passes=draft_calls, accepted_draft_tokens=accepted_total, proposed_draft_tokens=proposed_total, wall_seconds=wall, target_seconds=target_seconds, context_seconds=context_seconds, draft_seconds=draft_seconds, selection_seconds=selection_seconds, selector_pair_scores=markov_scores)
    meta = {"dspark_survival_floor": float(survival_floor), "dspark_markov_candidate_scores": int(markov_scores), "dspark_confidence_evaluations": int(confidence_evals), "dspark_trimmed_tokens": int(trimmed), "dspark_mean_confidence": float(confidence_sum / max(confidence_evals, 1))}
    return seq, stats, meta


def v10_decode(runtime: LfmDSparkRuntime, input_ids: np.ndarray, max_new_tokens: int, *, config: V10Config):
    seq = np.asarray(input_ids, dtype=np.int64).copy(); start_len = int(seq.size)
    target_calls = draft_calls = accepted_total = proposed_total = 0
    target_seconds = context_seconds = draft_seconds = selection_seconds = 0.0
    candidate_scores = uncertain = sampled = fast = 0; wall0 = time.perf_counter()
    while int(seq.size) - start_len < int(max_new_tokens):
        remaining = int(max_new_tokens) - (int(seq.size) - start_len)
        t = time.perf_counter(); context = runtime.context_features(seq); context_seconds += time.perf_counter() - t
        t = time.perf_counter(); draft_logits = runtime.draft_logits(context); draft_seconds += time.perf_counter() - t; draft_calls += 1
        t = time.perf_counter(); full, sel = select_v10_quickpath(runtime, draft_logits, context, int(seq[-1]), config)
        proposal = np.asarray(full[: min(int(full.size), remaining)], dtype=np.int64)
        candidate_scores += int(sel["candidate_scores"]); uncertain += int(sel["uncertain_positions"]); sampled += int(sel["sampled_positions"]); fast += int(sel["fast_argmax_positions"])
        selection_seconds += time.perf_counter() - t; proposed_total += int(proposal.size)
        verifier, accepted, elapsed = _verify(runtime, seq, proposal)
        target_seconds += elapsed; target_calls += 1; accepted_total += int(accepted)
        if accepted: seq = np.concatenate([seq, proposal[:accepted]])
        if accepted < int(proposal.size) and int(seq.size) - start_len < int(max_new_tokens): seq = np.append(seq, verifier[accepted])
    seq = seq[: start_len + int(max_new_tokens)]; wall = time.perf_counter() - wall0
    stats = RealDecodeStats(method="boltzmann_v10", new_tokens=int(max_new_tokens), target_forward_passes=target_calls, draft_forward_passes=draft_calls, accepted_draft_tokens=accepted_total, proposed_draft_tokens=proposed_total, wall_seconds=wall, target_seconds=target_seconds, context_seconds=context_seconds, draft_seconds=draft_seconds, selection_seconds=selection_seconds, boltzmann_candidate_scores=candidate_scores)
    meta = {"v10_candidate_scores": int(candidate_scores), "v10_uncertain_positions": int(uncertain), "v10_sampled_positions": int(sampled), "v10_fast_argmax_positions": int(fast)}
    return seq, stats, meta


def _aggregate_trials(rows: list[dict], tokens: int) -> dict:
    wall = sum(float(x["wall_seconds"]) for x in rows); total_tokens = int(tokens) * len(rows)
    return {"tokens_per_second": float(total_tokens / max(wall, 1e-12)), "mean_acceptance_rate": float(statistics.fmean(float(x["acceptance_rate"]) for x in rows)), "mean_target_forward_passes": float(statistics.fmean(float(x["target_forward_passes"]) for x in rows)), "all_exact": bool(all(bool(x["exact_match"]) for x in rows))}


def calibrate_dspark(runtime: LfmDSparkRuntime, prompts: list[str], *, tokens: int, top_k: int):
    references = {p: _greedy_reference(runtime, runtime.encode(p), tokens) for p in prompts}; candidates = []
    for floor in DSPARK_FLOORS:
        rows = []
        for p in prompts:
            ids = runtime.encode(p); out, stats, meta = dspark_decode(runtime, ids, tokens, top_k=top_k, survival_floor=floor)
            row = stats.to_dict(); row.update(meta); row["exact_match"] = bool(np.array_equal(out, references[p])); rows.append(row)
        agg = _aggregate_trials(rows, tokens); agg["survival_floor"] = float(floor); candidates.append(agg)
    valid = [x for x in candidates if x["all_exact"]]
    if not valid: raise RuntimeError("No exact DSpark-Lite calibration setting")
    best = max(valid, key=lambda x: (x["tokens_per_second"], -x["mean_target_forward_passes"]))
    return float(best["survival_floor"]), candidates


def _eval_v10_configs(runtime: LfmDSparkRuntime, prompts: list[str], references: dict[str, np.ndarray], configs, tokens: int):
    results = []
    for cfg in configs:
        rows = []
        for p in prompts:
            ids = runtime.encode(p); out, stats, meta = v10_decode(runtime, ids, tokens, config=cfg)
            row = stats.to_dict(); row.update(meta); row["exact_match"] = bool(np.array_equal(out, references[p])); rows.append(row)
        agg = _aggregate_trials(rows, tokens); agg["config"] = config_dict(cfg); agg["config_key"] = config_key(cfg); agg["mean_candidate_scores"] = float(statistics.fmean(float(x["v10_candidate_scores"]) for x in rows)); results.append(agg)
    return results


def calibrate_v10(runtime: LfmDSparkRuntime, prompts: list[str], *, tokens: int):
    grid = v10_grid(); plan = successive_halving_plan(len(grid), len(prompts), survivors=4, stage1_prompts=2)
    references = {p: _greedy_reference(runtime, runtime.encode(p), tokens) for p in prompts}
    stage1_prompts = prompts[: plan["stage1_prompts"]]; stage1 = _eval_v10_configs(runtime, stage1_prompts, references, grid, tokens)
    stage1_valid = [x for x in stage1 if x["all_exact"]]; stage1_valid.sort(key=lambda x: x["tokens_per_second"], reverse=True)
    survivor_keys = {x["config_key"] for x in stage1_valid[: plan["survivors"]]}; survivors = [cfg for cfg in grid if config_key(cfg) in survivor_keys]
    stage2 = _eval_v10_configs(runtime, prompts, references, survivors, tokens); valid = [x for x in stage2 if x["all_exact"]]
    if not valid: raise RuntimeError("No exact V10 calibration setting")
    best = max(valid, key=lambda x: (x["tokens_per_second"], -x["mean_candidate_scores"])); selected = next(cfg for cfg in survivors if config_key(cfg) == best["config_key"])
    return selected, {"plan": plan, "stage1": stage1, "stage2": stage2, "selected": best}


def _summary(rows: list[dict]) -> dict:
    def mean(key): return float(statistics.fmean(float(x[key]) for x in rows))
    def median(key): return float(statistics.median(float(x[key]) for x in rows))
    return {"tokens_per_second_median": median("tokens_per_second"), "latency_seconds_median": median("wall_seconds"), "target_seconds_median": median("target_seconds"), "draft_seconds_median": median("draft_seconds"), "selection_seconds_median": median("selection_seconds"), "mean_target_forward_passes": mean("target_forward_passes"), "mean_draft_forward_passes": mean("draft_forward_passes"), "mean_acceptance_rate": mean("acceptance_rate"), "mean_tokens_per_target_pass": mean("tokens_per_target_pass"), "mean_total_guidance_scores": mean("total_guidance_scores"), "all_exact": all(bool(x["exact_match"]) for x in rows)}


def benchmark_new_methods(runtime: LfmDSparkRuntime, prompts: list[str], *, tokens: int, repeats: int, top_k: int, dspark_floor: float, v10_config: V10Config):
    rows = {"dspark_v9": [], "boltzmann_v10": []}; outputs = []
    for pi, prompt in enumerate(prompts):
        ids = runtime.encode(prompt); reference = _greedy_reference(runtime, ids, tokens); first = {}
        for repeat in range(max(1, int(repeats))):
            order = ["dspark_v9", "boltzmann_v10"] if (pi + repeat) % 2 == 0 else ["boltzmann_v10", "dspark_v9"]
            for method in order:
                if method == "dspark_v9": out, stats, meta = dspark_decode(runtime, ids, tokens, top_k=top_k, survival_floor=dspark_floor)
                else: out, stats, meta = v10_decode(runtime, ids, tokens, config=v10_config)
                exact = bool(np.array_equal(out, reference)); row = stats.to_dict(); row.update(meta); row.update(prompt=prompt, prompt_index=pi, repeat=repeat, exact_match=exact); rows[method].append(row); first.setdefault(method, runtime.decode(out))
        outputs.append({"prompt": prompt, "normal": runtime.decode(reference), **first})
    return rows, outputs


def build_report(data: dict, path: str | Path) -> None:
    summary = data["summary"]; winner = data["winner"]; methods = [m for m in ALL_METHODS if m in summary]; rows = []
    for m in sorted(methods, key=lambda x: summary[x]["tokens_per_second_median"], reverse=True):
        s = summary[m]; cls = "win" if m == winner else ("new" if m in EXTRA_LABELS else "")
        rows.append(f"<tr class='{cls}'><th>{html.escape(ALL_LABELS[m])}</th><td>{s['tokens_per_second_median']:.3f}</td><td>{s['speedup_vs_normal']:.3f}×</td><td>{100*s['mean_acceptance_rate']:.1f}%</td><td>{s['mean_tokens_per_target_pass']:.2f}</td><td>{s['mean_target_forward_passes']:.2f}</td><td>{s['selection_seconds_median']:.4f}s</td><td>{s['mean_total_guidance_scores']:.1f}</td><td>{'✓' if s['all_exact'] else '✗'}</td></tr>")
    ds = data["dspark_v9"]; v10 = data["boltzmann_v10"]; ds_acc = ds["training"].get("teacher_conditioned_position_accuracy", []); ds_acc_text = ", ".join(f"{100*x:.1f}%" for x in ds_acc)
    stage2_rows = "".join(f"<tr><td>{html.escape(x['config_key'])}</td><td>{x['tokens_per_second']:.3f}</td><td>{100*x['mean_acceptance_rate']:.1f}%</td><td>{x['mean_candidate_scores']:.1f}</td></tr>" for x in sorted(v10["calibration"]["stage2"], key=lambda z: z["tokens_per_second"], reverse=True))
    js = json.dumps({m: {"label": ALL_LABELS[m], "tps": summary[m]["tokens_per_second_median"], "accept": summary[m]["mean_acceptance_rate"]} for m in methods}, separators=(",", ":"))
    old_scores = float(summary["dflash6_boltzmann"]["mean_total_guidance_scores"]); new_scores = float(summary["boltzmann_v10"]["mean_total_guidance_scores"]); reduction = 1.0 - new_scores / max(old_scores, 1e-9)
    css = ":root{font-family:Inter,ui-sans-serif,system-ui;background:#07101d;color:#eef7ff;--panel:#0d1b2c;--line:#24405b;--cyan:#54d7ff;--green:#7bf2b3;--muted:#98abc0}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 10% 0,#143754,transparent 35%),#07101d}.wrap{max-width:1240px;margin:auto;padding:28px 20px 70px}.hero,.card,.demo{background:linear-gradient(145deg,#102339,#0a1727);border:1px solid var(--line);border-radius:22px;padding:24px;margin-bottom:20px}.hero h1{font-size:clamp(2.2rem,5vw,4.4rem);line-height:1;margin:.2em 0}.chips{display:flex;gap:8px;flex-wrap:wrap}.chip{border:1px solid #355774;border-radius:999px;padding:6px 10px;color:#bcecff}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.metric{background:#091626;border:1px solid var(--line);padding:14px;border-radius:16px}.metric b{display:block;font-size:1.5rem;color:var(--green)}.muted{color:var(--muted)}table{width:100%;border-collapse:collapse;min-width:900px}th,td{padding:10px;border-bottom:1px solid #1f354b;text-align:right}th:first-child{text-align:left}.table{overflow:auto}.win{background:#123324}.new{background:#101f37}.bar{height:16px;background:#07111d;border:1px solid #29455f;border-radius:999px;overflow:hidden}.fill{height:100%;background:linear-gradient(90deg,var(--cyan),var(--green));width:0;transition:width .5s}.race{display:grid;grid-template-columns:180px 1fr 60px;gap:10px;align-items:center;margin:12px 0}select,button{background:#10263a;color:#eef7ff;border:1px solid #365875;border-radius:10px;padding:9px}.tokens{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}.tok{padding:9px 10px;border-radius:10px;background:#10253a;border:1px solid #365874}.tok.a{background:#123a2e;border-color:#3ab87f}.tok.r{background:#3b1d28;border-color:#d75f77}.flow{display:grid;grid-template-columns:repeat(6,1fr);gap:9px}.node{min-height:75px;border:1px solid #2b4a66;border-radius:14px;display:grid;place-items:center;text-align:center;padding:8px}.node.on{box-shadow:0 0 24px #54d7ff44;border-color:var(--cyan)}@media(max-width:850px){.grid{grid-template-columns:1fr}.metrics{grid-template-columns:1fr 1fr}.flow{grid-template-columns:1fr 1fr}}"
    doc = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>LFM2.5 V9 DSpark + V10 Boltzmann</title><style>{css}</style></head><body><main class='wrap'>
<section class='hero'><div class='chips'><span class='chip'>LFM2.5-350M-Base</span><span class='chip'>CPU · float32 · 2 threads</span><span class='chip'>11 methods</span><span class='chip'>exact greedy verification</span></div><h1>Can V9 or V10 beat DFlash?</h1><p class='muted'>V9 adds DSpark-style low-rank Markov dependency and confidence scheduling to the frozen DFlash backbone. V10 keeps Boltzmann training-free and spends stochastic work only on a tiny uncertainty budget.</p><div class='metrics'><div class='metric'><small>Winner</small><b>{html.escape(ALL_LABELS[winner])}</b></div><div class='metric'><small>Winner speed</small><b>{summary[winner]['speedup_vs_normal']:.3f}×</b></div><div class='metric'><small>V10 stochastic-work reduction</small><b>{100*reduction:.1f}%</b></div><div class='metric'><small>Exact methods</small><b>{sum(int(summary[m]['all_exact']) for m in methods)}/{len(methods)}</b></div></div></section>
<section class='card'><h2>Measured ranking</h2><div class='table'><table><thead><tr><th>Method</th><th>tok/s</th><th>vs Normal</th><th>Acceptance</th><th>Tokens/pass</th><th>Target passes</th><th>Select time</th><th>Guidance ops</th><th>Exact</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<section class='grid'><div class='card'><h2>V9 · DSpark-Lite</h2><p>Frozen DFlash backbone → low-rank Markov correction → learned prefix-survival confidence → exact target verification.</p><p><b>Markov rank:</b> {ds['training']['markov_rank']} · <b>training steps:</b> {ds['training']['markov_steps']} + {ds['training']['confidence_steps']} · <b>selected survival floor:</b> {ds['selected_survival_floor']:g}</p><p class='muted'>Teacher-conditioned position accuracy after Markov correction: {html.escape(ds_acc_text)}</p></div><div class='card'><h2>V10 · Advanced Boltzmann QuickPath</h2><p>Top-2 only, deterministic argmax on confident positions, and at most {v10['selected_config']['stochastic_budget']} stochastic slot(s) per draft block.</p><p><b>Selected:</b> {html.escape(config_key(V10Config(**v10['selected_config'])))}</p><p><b>Training:</b> 0 steps. Successive-halving calibration used {v10['calibration']['plan']['successive_halving_prompt_config_evaluations']} prompt-config evaluations instead of {v10['calibration']['plan']['full_grid_prompt_config_evaluations']} ({100*v10['calibration']['plan']['calibration_work_reduction_fraction']:.1f}% less).</p></div></section>
<section class='card'><h2>Animated architecture</h2><div class='flow'><div class='node'>Prompt/context</div><div class='node'>Parallel DFlash draft</div><div class='node'>V9 Markov / V10 gate</div><div class='node'>Confidence / logistic coin</div><div class='node'>LFM target verify</div><div class='node'>Accept + correction</div></div></section>
<section class='grid'><div class='demo'><h2>Live demo 1 · measured race</h2><p class='muted'>Choose two methods. Browser time is accelerated; the ratio is driven by measured tok/s.</p><p><select id='a'></select> <select id='b'></select> <button id='go'>Race 48 tokens</button></p><div class='race'><b id='al'></b><div class='bar'><div class='fill' id='af'></div></div><span id='ao'>0</span></div><div class='race'><b id='bl'></b><div class='bar'><div class='fill' id='bf'></div></div><span id='bo'>0</span></div><p id='verdict'></p></div>
<div class='demo'><h2>Live demo 2 · selection budget</h2><p><select id='m'><option value='dflash'>DFlash</option><option value='dspark_v9'>V9 DSpark-Lite</option><option value='dflash6_boltzmann'>DFlash6-Boltzmann</option><option value='boltzmann_v10'>V10 Advanced Boltzmann</option></select> <button id='animate'>Animate block</button></p><div class='tokens' id='tokens'></div><p class='muted' id='note'>This is a data-driven mechanism visualization, not browser-side model inference.</p></div></section>
<section class='card'><h2>V10 final calibration survivors</h2><div class='table'><table><thead><tr><th>Configuration</th><th>tok/s</th><th>Acceptance</th><th>candidate scores/run</th></tr></thead><tbody>{stage2_rows}</tbody></table></div></section>
<section class='card'><h2>Interpretation</h2><p>The decisive comparison is against plain DFlash and DFlash6-Boltzmann, not only Normal. V9 is useful only if its extra Markov/confidence work buys enough additional accepted prefix. V10 is useful only if it keeps most of Boltzmann's proposal-quality gain while removing its top-k/Gumbel overhead. Every accepted output remains target-authoritative.</p></section>
</main><script>const D={js};const methods=Object.keys(D),$=x=>document.getElementById(x);function opts(id,sel){{$(id).innerHTML=methods.map(m=>`<option value="${{m}}" ${{m===sel?'selected':''}}>${{D[m].label}}</option>`).join('')}}opts('a','dflash');opts('b','boltzmann_v10');let timer=null;$('go').onclick=()=>{{clearInterval(timer);const a=$('a').value,b=$('b').value;$('al').textContent=D[a].label;$('bl').textContent=D[b].label;const total=48,start=performance.now(),scale=.16/Math.max(D[a].tps,D[b].tps);timer=setInterval(()=>{{const e=(performance.now()-start)/1000,ta=Math.min(total,e*D[a].tps/scale),tb=Math.min(total,e*D[b].tps/scale);$('af').style.width=(100*ta/total)+'%';$('bf').style.width=(100*tb/total)+'%';$('ao').textContent=Math.floor(ta);$('bo').textContent=Math.floor(tb);if(ta>=total||tb>=total){{clearInterval(timer);const w=ta>=total?a:b;$('verdict').textContent=D[w].label+' wins using the measured throughput ratio.'}}}},40)}};$('animate').onclick=()=>{{const m=$('m').value,x=D[m];$('tokens').innerHTML='';const n=4;let rejected=false;for(let i=0;i<n;i++){{const e=document.createElement('span');e.className='tok';e.textContent='slot '+(i+1);$('tokens').appendChild(e);setTimeout(()=>{{const seed=((i+1)*37 + Math.round(x.accept*1000))%1000/1000;const ok=!rejected&&seed<x.accept;if(ok)e.classList.add('a');else{{rejected=true;e.classList.add('r')}}}},250+220*i)}}$('note').textContent=x.label+' · measured acceptance '+(100*x.accept).toFixed(1)+'%';document.querySelectorAll('.node').forEach((n,i)=>setTimeout(()=>{{n.classList.add('on');setTimeout(()=>n.classList.remove('on'),350)}},i*300))}};</script></body></html>"""
    Path(path).write_text(doc, encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    base = run_existing_showcase(args)
    runtime = LfmDSparkRuntime(args.aux, args.dspark, model_id=args.model_id, cpu_threads=args.cpu_threads, dtype=args.dtype)
    benchmark_prompts = _read_prompts(args.prompts, args.prompt_limit); calibration_prompts = _read_prompts(args.calibration_prompts, args.calibration_prompt_limit)
    warm = runtime.encode(calibration_prompts[0]); _ = runtime.target_logits(warm); _ = runtime.draft_logits(runtime.context_features(warm))
    dspark_floor, dspark_cal = calibrate_dspark(runtime, calibration_prompts, tokens=args.calibration_tokens, top_k=args.top_k)
    v10_config, v10_cal = calibrate_v10(runtime, calibration_prompts, tokens=args.calibration_tokens)
    new_rows, new_outputs = benchmark_new_methods(runtime, benchmark_prompts, tokens=args.tokens, repeats=args.repeats, top_k=args.top_k, dspark_floor=dspark_floor, v10_config=v10_config)
    base["runs"].extend(new_rows["dspark_v9"] + new_rows["boltzmann_v10"])
    for method in ("dspark_v9", "boltzmann_v10"): base["summary"][method] = _summary(new_rows[method])
    baseline = float(base["summary"]["normal"]["tokens_per_second_median"])
    for method in ("dspark_v9", "boltzmann_v10"): base["summary"][method]["speedup_vs_normal"] = float(base["summary"][method]["tokens_per_second_median"] / max(baseline, 1e-12))
    base["summary"]["dspark_v9"]["mean_markov_candidate_scores"] = float(statistics.fmean(x["dspark_markov_candidate_scores"] for x in new_rows["dspark_v9"])); base["summary"]["dspark_v9"]["mean_confidence_evaluations"] = float(statistics.fmean(x["dspark_confidence_evaluations"] for x in new_rows["dspark_v9"])); base["summary"]["boltzmann_v10"]["mean_v10_candidate_scores"] = float(statistics.fmean(x["v10_candidate_scores"] for x in new_rows["boltzmann_v10"])); base["summary"]["boltzmann_v10"]["mean_fast_argmax_positions"] = float(statistics.fmean(x["v10_fast_argmax_positions"] for x in new_rows["boltzmann_v10"]))
    base["dspark_v9"] = {"selected_survival_floor": float(dspark_floor), "calibration": dspark_cal, "training": runtime.dspark_metadata.get("training", {}), "parameter_count": int(runtime.dspark_parameter_count), "mechanism": runtime.dspark_metadata.get("mechanism"), "inference_candidate_rescoring": runtime.dspark_metadata.get("inference_candidate_rescoring")}
    base["boltzmann_v10"] = {"selected_config": config_dict(v10_config), "calibration": v10_cal, "training_steps": 0, "mechanism": "Top-2 uncertainty-gated exact logistic Boltzmann sampling with a fixed stochastic budget; all other positions use DFlash argmax."}
    base["showcase"]["version"] = 10; base["showcase"]["live_demo_count"] = 2; base["showcase"]["new_outputs"] = new_outputs
    base["config"]["benchmark_workload"] = f"{len(benchmark_prompts)} held-out prompts × {args.tokens} generated tokens × {args.repeats} repeats"
    base["method_scope"] = "Same frozen LiquidAI/LFM2.5-350M-Base verifier. V9 DSpark-Lite adds only low-rank Markov + confidence heads to the frozen DFlash backbone. V10 Advanced Boltzmann is training-free and uses top-2 uncertainty-gated logistic sampling. Exact greedy target verification is authoritative for every method."
    base["winner"] = max(ALL_METHODS, key=lambda m: base["summary"][m]["tokens_per_second_median"])
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True); (out / "benchmark.json").write_text(json.dumps(base, indent=2), encoding="utf-8"); build_report(base, out / "report.html"); return base


def main() -> None:
    parser = argparse.ArgumentParser(description="LFM2.5 V9 DSpark-Lite + V10 Advanced Boltzmann study")
    parser.add_argument("--aux", default="lfm-artifacts/lfm_aux.pt"); parser.add_argument("--dspark", default="lfm-artifacts/lfm_dspark.pt"); parser.add_argument("--prompts", default="real_benchmarks/prompts.json"); parser.add_argument("--calibration-prompts", default="real_benchmarks/calibration_prompts.json"); parser.add_argument("--output-dir", default="lfm-reports"); parser.add_argument("--model-id", default="LiquidAI/LFM2.5-350M-Base")
    parser.add_argument("--tokens", type=int, default=24); parser.add_argument("--repeats", type=int, default=3); parser.add_argument("--prompt-limit", type=int, default=6); parser.add_argument("--calibration-tokens", type=int, default=8); parser.add_argument("--calibration-prompt-limit", type=int, default=4); parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--jump-weight", type=float, default=0.5); parser.add_argument("--fused-weight", type=float, default=1.0); parser.add_argument("--fused-min-margin", type=float, default=0.0); parser.add_argument("--boltzmann-temperature", type=float, default=0.15); parser.add_argument("--bmobs-temperature", type=float, default=0.35); parser.add_argument("--cpu-threads", type=int, default=2); parser.add_argument("--dtype", choices=("float32",), default="float32")
    args = parser.parse_args(); payload = run(args); print(json.dumps({"winner": payload["winner"], "summary": {k: payload["summary"][k] for k in ALL_METHODS}, "dspark_floor": payload["dspark_v9"]["selected_survival_floor"], "v10": payload["boltzmann_v10"]["selected_config"]}, indent=2))


if __name__ == "__main__": main()
