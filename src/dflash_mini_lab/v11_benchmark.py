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
from .lfm_v9v10_benchmark import _greedy_reference, dspark_decode, run as run_v10_study, v10_decode
from .lfm_v10 import V10Config
from .v11_boltzmann_mobs import V11Config, config_dict, config_key, select_v11_boltzmann_gated_mobs, v11_grid


V11_METHOD = "boltzmann_gated_mobs_v11"
FOCUS_METHODS = ("normal", "dflash", "dflash3_mobs", "dspark_v9", "boltzmann_v10", V11_METHOD)
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
    t = time.perf_counter(); logits = runtime.target_logits(verify_input); elapsed = time.perf_counter() - t
    p, k = int(seq.size), int(proposal.size)
    verifier = np.argmax(logits[p - 1 : p - 1 + k], axis=-1).astype(np.int64)
    mismatch = np.flatnonzero(proposal != verifier)
    accepted = k if mismatch.size == 0 else int(mismatch[0])
    return verifier, accepted, elapsed


def v11_decode(runtime: LfmDSparkRuntime, input_ids: np.ndarray, max_new_tokens: int, *, config: V11Config):
    seq = np.asarray(input_ids, dtype=np.int64).copy(); start_len = int(seq.size)
    target_calls = draft_calls = accepted_total = proposed_total = 0
    target_seconds = context_seconds = draft_seconds = selection_seconds = 0.0
    pair_scores = gate_evals = gated = eligible = fast = 0; uncertainty_sum = 0.0
    wall0 = time.perf_counter()
    while int(seq.size) - start_len < int(max_new_tokens):
        remaining = int(max_new_tokens) - (int(seq.size) - start_len)
        t = time.perf_counter(); context = runtime.context_features(seq); context_seconds += time.perf_counter() - t
        t = time.perf_counter(); draft_logits = runtime.draft_logits(context); draft_seconds += time.perf_counter() - t; draft_calls += 1
        t = time.perf_counter(); full, meta = select_v11_boltzmann_gated_mobs(runtime, draft_logits, context, int(seq[-1]), config)
        proposal = np.asarray(full[: min(int(full.size), remaining)], dtype=np.int64)
        block_evals = min(int(full.size), remaining); gate_evals += block_evals
        pair_scores += int(meta["pair_scores"]); gated += int(meta["gated_positions"]); eligible += int(meta["eligible_positions"]); fast += int(meta["fast_argmax_positions"])
        uncertainty_sum += float(meta.get("mean_uncertainty", 0.0)) * block_evals; selection_seconds += time.perf_counter() - t; proposed_total += int(proposal.size)
        verifier, accepted, elapsed = _verify(runtime, seq, proposal); target_seconds += elapsed; target_calls += 1; accepted_total += int(accepted)
        if accepted: seq = np.concatenate([seq, proposal[:accepted]])
        if accepted < int(proposal.size) and int(seq.size) - start_len < int(max_new_tokens): seq = np.append(seq, verifier[accepted])
    seq = seq[: start_len + int(max_new_tokens)]; wall = time.perf_counter() - wall0
    stats = RealDecodeStats(method=V11_METHOD, new_tokens=int(max_new_tokens), target_forward_passes=target_calls, draft_forward_passes=draft_calls, accepted_draft_tokens=accepted_total, proposed_draft_tokens=proposed_total, wall_seconds=wall, target_seconds=target_seconds, context_seconds=context_seconds, draft_seconds=draft_seconds, selection_seconds=selection_seconds, selector_pair_scores=pair_scores, boltzmann_candidate_scores=gate_evals)
    meta = {"v11_pair_scores": int(pair_scores), "v11_gate_evaluations": int(gate_evals), "v11_gated_positions": int(gated), "v11_eligible_positions": int(eligible), "v11_fast_argmax_positions": int(fast), "v11_mean_uncertainty": float(uncertainty_sum / max(gate_evals, 1))}
    return seq, stats, meta


def _aggregate(rows: list[dict], tokens: int) -> dict:
    wall = sum(float(x["wall_seconds"]) for x in rows); total = int(tokens) * len(rows)
    return {"tokens_per_second": float(total / max(wall, 1e-12)), "mean_acceptance_rate": float(statistics.fmean(float(x["acceptance_rate"]) for x in rows)), "mean_target_forward_passes": float(statistics.fmean(float(x["target_forward_passes"]) for x in rows)), "mean_guidance_scores": float(statistics.fmean(float(x["total_guidance_scores"]) for x in rows)), "all_exact": bool(all(bool(x["exact_match"]) for x in rows))}


def _eval_configs(runtime, prompts, references, configs, tokens):
    results = []
    for cfg in configs:
        rows = []
        for prompt in prompts:
            ids = runtime.encode(prompt); out, stats, meta = v11_decode(runtime, ids, tokens, config=cfg)
            row = stats.to_dict(); row.update(meta); row["exact_match"] = bool(np.array_equal(out, references[prompt])); rows.append(row)
        agg = _aggregate(rows, tokens); agg.update(config=config_dict(cfg), config_key=config_key(cfg), mean_pair_scores=float(statistics.fmean(float(x["v11_pair_scores"]) for x in rows)), mean_gated_positions=float(statistics.fmean(float(x["v11_gated_positions"]) for x in rows))); results.append(agg)
    return results


def calibrate_v11(runtime: LfmDSparkRuntime, prompts: list[str], *, tokens: int):
    grid = v11_grid(); references = {p: _greedy_reference(runtime, runtime.encode(p), tokens) for p in prompts}
    stage1 = _eval_configs(runtime, prompts[:1], references, grid, tokens)
    valid1 = sorted((x for x in stage1 if x["all_exact"]), key=lambda x: (x["tokens_per_second"], -x["mean_guidance_scores"]), reverse=True)
    survivor_keys = {x["config_key"] for x in valid1[:4]}; survivors = [cfg for cfg in grid if config_key(cfg) in survivor_keys]
    stage2 = _eval_configs(runtime, prompts, references, survivors, tokens); valid2 = [x for x in stage2 if x["all_exact"]]
    if not valid2: raise RuntimeError("No exact V11 calibration setting")
    best = max(valid2, key=lambda x: (x["tokens_per_second"], -x["mean_guidance_scores"])); selected = next(cfg for cfg in survivors if config_key(cfg) == best["config_key"])
    full = len(grid) * len(prompts); used = len(grid) + len(survivors) * len(prompts)
    return selected, {"stage1": stage1, "stage2": stage2, "selected": best, "plan": {"total_configs": len(grid), "stage1_prompts": 1, "survivors": len(survivors), "full_grid_prompt_config_evaluations": full, "successive_halving_prompt_config_evaluations": used, "calibration_work_reduction_fraction": 1.0 - used / max(full, 1), "training_steps": 0}}


def _summary(rows: list[dict]) -> dict:
    mean = lambda k: float(statistics.fmean(float(x[k]) for x in rows)); median = lambda k: float(statistics.median(float(x[k]) for x in rows))
    return {"tokens_per_second_median": median("tokens_per_second"), "latency_seconds_median": median("wall_seconds"), "target_seconds_median": median("target_seconds"), "draft_seconds_median": median("draft_seconds"), "selection_seconds_median": median("selection_seconds"), "mean_target_forward_passes": mean("target_forward_passes"), "mean_draft_forward_passes": mean("draft_forward_passes"), "mean_acceptance_rate": mean("acceptance_rate"), "mean_tokens_per_target_pass": mean("tokens_per_target_pass"), "mean_total_guidance_scores": mean("total_guidance_scores"), "all_exact": bool(all(bool(x["exact_match"]) for x in rows))}


def benchmark_focus(runtime, prompts, *, tokens, repeats, top_k, dspark_floor, v10_config, v11_config):
    rows = {m: [] for m in FOCUS_METHODS}; outputs = []
    for pi, prompt in enumerate(prompts):
        ids = runtime.encode(prompt); reference = _greedy_reference(runtime, ids, tokens); first = {}
        for repeat in range(max(1, int(repeats))):
            shift = (pi * max(1, int(repeats)) + repeat) % len(FOCUS_METHODS); order = FOCUS_METHODS[shift:] + FOCUS_METHODS[:shift]
            for method in order:
                if method == "normal": out, stats = normal_decode(runtime, ids, tokens); meta = {}
                elif method == "dflash": out, stats = speculative_decode(runtime, ids, tokens, "dflash", top_k=top_k); meta = {}
                elif method == "dflash3_mobs": out, stats = speculative_decode(runtime, ids, tokens, "dflash3_mobs", top_k=top_k); meta = {}
                elif method == "dspark_v9": out, stats, meta = dspark_decode(runtime, ids, tokens, top_k=top_k, survival_floor=dspark_floor)
                elif method == "boltzmann_v10": out, stats, meta = v10_decode(runtime, ids, tokens, config=v10_config)
                else: out, stats, meta = v11_decode(runtime, ids, tokens, config=v11_config)
                row = stats.to_dict(); row.update(meta); row.update(prompt=prompt, prompt_index=pi, repeat=repeat, exact_match=bool(np.array_equal(out, reference))); rows[method].append(row); first.setdefault(method, runtime.decode(out))
        outputs.append({"prompt": prompt, "reference": runtime.decode(reference), "texts": first})
    summary = {m: _summary(rows[m]) for m in FOCUS_METHODS}; normal_tps = float(summary["normal"]["tokens_per_second_median"])
    for method in FOCUS_METHODS: summary[method]["speedup_vs_normal"] = float(summary[method]["tokens_per_second_median"] / max(normal_tps, 1e-12))
    v = summary[V11_METHOD]; v["mean_pair_scores"] = float(statistics.fmean(float(x["v11_pair_scores"]) for x in rows[V11_METHOD])); v["mean_gate_evaluations"] = float(statistics.fmean(float(x["v11_gate_evaluations"]) for x in rows[V11_METHOD])); v["mean_gated_positions"] = float(statistics.fmean(float(x["v11_gated_positions"]) for x in rows[V11_METHOD])); v["mean_fast_argmax_positions"] = float(statistics.fmean(float(x["v11_fast_argmax_positions"]) for x in rows[V11_METHOD]))
    return rows, summary, outputs


def build_report(data: dict, path: str | Path) -> None:
    focus = data["v11"]["focus_summary"]; winner = data["v11"]["focus_winner"]; ordered = sorted(FOCUS_METHODS, key=lambda m: focus[m]["tokens_per_second_median"], reverse=True)
    rows = "".join(f"<tr class='{'win' if m==winner else ''}'><th>{html.escape(LABELS[m])}</th><td>{focus[m]['tokens_per_second_median']:.3f}</td><td>{focus[m]['speedup_vs_normal']:.3f}×</td><td>{100*focus[m]['mean_acceptance_rate']:.1f}%</td><td>{focus[m]['mean_tokens_per_target_pass']:.2f}</td><td>{focus[m]['selection_seconds_median']:.4f}s</td><td>{focus[m]['mean_total_guidance_scores']:.1f}</td><td>{'✓' if focus[m]['all_exact'] else '✗'}</td></tr>" for m in ordered)
    cfg = data["v11"]["selected_config"]; v = focus[V11_METHOD]; mobs_ops = float(focus["dflash3_mobs"]["mean_total_guidance_scores"]); op_reduction = 1.0 - float(v["mean_total_guidance_scores"]) / max(mobs_ops, 1e-9); lead = float(data["v11"]["lead_vs_next_fraction"])
    finalists = "".join(f"<tr><td>{html.escape(x['config_key'])}</td><td>{x['tokens_per_second']:.3f}</td><td>{100*x['mean_acceptance_rate']:.1f}%</td><td>{x['mean_guidance_scores']:.1f}</td></tr>" for x in sorted(data["v11"]["calibration"]["stage2"], key=lambda z: z["tokens_per_second"], reverse=True))
    js = json.dumps({m:{"label":LABELS[m],"tps":focus[m]["tokens_per_second_median"]} for m in FOCUS_METHODS}, separators=(",",":"))
    css = ":root{font-family:Inter,system-ui;background:#07111f;color:#eef7ff;--line:#294963;--cyan:#58d9ff;--green:#7cf5b2;--muted:#98abc0}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% 0,#173b5a,transparent 35%),#07111f}.wrap{max-width:1200px;margin:auto;padding:28px 20px 70px}.hero,.card,.demo{background:linear-gradient(145deg,#10253b,#0a1727);border:1px solid var(--line);border-radius:22px;padding:24px;margin-bottom:20px}h1{font-size:clamp(2.2rem,5vw,4.4rem);line-height:.97}.chips{display:flex;gap:8px;flex-wrap:wrap}.chip{border:1px solid #355a78;border-radius:999px;padding:6px 10px}.metrics,.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.grid{grid-template-columns:1fr 1fr}.metric{border:1px solid var(--line);padding:14px;border-radius:15px}.metric b{display:block;font-size:1.5rem;color:var(--green)}.muted{color:var(--muted)}.table{overflow:auto}table{width:100%;border-collapse:collapse;min-width:850px}th,td{padding:10px;border-bottom:1px solid #20384f;text-align:right}th:first-child{text-align:left}.win{background:#123a29}.flow{display:grid;grid-template-columns:repeat(6,1fr);gap:8px}.node{min-height:78px;border:1px solid #31506b;border-radius:13px;display:grid;place-items:center;text-align:center;padding:8px}.node.on{border-color:var(--cyan);box-shadow:0 0 24px #58d9ff44}.race{display:grid;grid-template-columns:180px 1fr 50px;gap:8px;align-items:center;margin:10px 0}.track{height:18px;border:1px solid #29475f;border-radius:999px;overflow:hidden}.fill{height:100%;width:0;background:linear-gradient(90deg,var(--cyan),var(--green))}.slots{display:flex;gap:8px;flex-wrap:wrap}.slot{padding:11px;border:1px solid #365773;border-radius:10px}.fast{background:#12362b}.mobs{background:#3b3013}select,button{background:#10263b;color:#eef7ff;border:1px solid #365b78;border-radius:10px;padding:9px}@media(max-width:850px){.metrics,.grid{grid-template-columns:1fr 1fr}.flow{grid-template-columns:1fr 1fr}}"
    doc = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>V11 Boltzmann-Gated MOBS</title><style>{css}</style></head><body><main class='wrap'><section class='hero'><div class='chips'><span class='chip'>LFM2.5-350M</span><span class='chip'>CPU float32 · 2 threads</span><span class='chip'>rotated order</span><span class='chip'>exact output</span></div><h1>V11 · Boltzmann-Gated MOBS</h1><p class='muted'>Cheap Boltzmann uncertainty routes only weak DFlash slots through MOBS. Confident slots keep the zero-guidance argmax path.</p><div class='metrics'><div class='metric'><small>Winner</small><b>{html.escape(LABELS[winner])}</b></div><div class='metric'><small>Winner speed</small><b>{focus[winner]['tokens_per_second_median']:.3f} tok/s</b></div><div class='metric'><small>Lead vs #2</small><b>{100*lead:.2f}%</b></div><div class='metric'><small>V11 guidance vs MOBS</small><b>{100*op_reduction:.1f}% less</b></div></div></section><section class='card'><h2>Focused contender race</h2><p class='muted'>6 held-out prompts × 24 tokens × 3 repeats. Execution order rotates every trial.</p><div class='table'><table><thead><tr><th>Method</th><th>tok/s</th><th>vs Normal</th><th>Acceptance</th><th>Tokens/pass</th><th>Select</th><th>Guidance</th><th>Exact</th></tr></thead><tbody>{rows}</tbody></table></div></section><section class='grid'><div class='card'><h2>Selected configuration</h2><p><code>{html.escape(config_key(V11Config(**cfg)))}</code></p><p>No training. Held-out successive-halving calibration only.</p></div><div class='card'><h2>Measured routing work</h2><p><b>{v['mean_gated_positions']:.1f}</b> MOBS-gated positions/run<br><b>{v['mean_fast_argmax_positions']:.1f}</b> fast argmax positions/run<br><b>{v['mean_pair_scores']:.1f}</b> MOBS pair scores/run<br><b>{v['mean_gate_evaluations']:.1f}</b> Boltzmann gate evaluations/run</p></div></section><section class='card'><h2>Animated architecture</h2><div class='flow'><div class='node'>Context</div><div class='node'>DFlash parallel draft</div><div class='node'>Top-2 Boltzmann uncertainty</div><div class='node'>Weak-slot gate</div><div class='node'>Sparse MOBS</div><div class='node'>Exact target verify</div></div></section><section class='grid'><div class='demo'><h2>Live demo 1 · measured race</h2><p><select id='a'></select> <select id='b'></select> <button id='go'>Race</button></p><div class='race'><b id='al'></b><div class='track'><div id='af' class='fill'></div></div><span id='ao'>0</span></div><div class='race'><b id='bl'></b><div class='track'><div id='bf' class='fill'></div></div><span id='bo'>0</span></div><p id='verdict' class='muted'>Data-driven animation.</p></div><div class='demo'><h2>Live demo 2 · V11 gate</h2><button id='gate'>Animate block</button><div id='slots' class='slots' style='margin-top:14px'></div><p class='muted'>Green = fast DFlash; amber = MOBS.</p></div></section><section class='card'><h2>Held-out calibration finalists</h2><div class='table'><table><thead><tr><th>Config</th><th>tok/s</th><th>Acceptance</th><th>Guidance</th></tr></thead><tbody>{finalists}</tbody></table></div></section></main><script>const D={js},M=Object.keys(D),$=x=>document.getElementById(x);function opts(id,s){{$ (id).innerHTML=M.map(m=>`<option value="${{m}}" ${{m===s?'selected':''}}>${{D[m].label}}</option>`).join('')}}opts('a','dflash3_mobs');opts('b','{V11_METHOD}');let t;$('go').onclick=()=>{{clearInterval(t);const a=$('a').value,b=$('b').value,total=48,start=performance.now(),scale=.16/Math.max(D[a].tps,D[b].tps);$('al').textContent=D[a].label;$('bl').textContent=D[b].label;t=setInterval(()=>{{const e=(performance.now()-start)/1000,aa=Math.min(total,e*D[a].tps/scale),bb=Math.min(total,e*D[b].tps/scale);$('af').style.width=100*aa/total+'%';$('bf').style.width=100*bb/total+'%';$('ao').textContent=Math.floor(aa);$('bo').textContent=Math.floor(bb);if(aa>=total||bb>=total){{clearInterval(t);$('verdict').textContent=D[aa>=total?a:b].label+' wins at the measured ratio.'}}}},40)}};$('gate').onclick=()=>{{$('slots').innerHTML='';const n={int(cfg['mobs_budget'])};for(let i=0;i<4;i++){{const e=document.createElement('span');e.className='slot';e.textContent='slot '+(i+1);$('slots').appendChild(e);setTimeout(()=>{{const k=i<n?'mobs':'fast';e.classList.add(k);e.textContent+=' · '+(k==='mobs'?'MOBS':'fast')}},220*i)}}document.querySelectorAll('.node').forEach((n,i)=>setTimeout(()=>{{n.classList.add('on');setTimeout(()=>n.classList.remove('on'),300)}},i*250))}};</script></body></html>"""
    Path(path).write_text(doc, encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    base = run_v10_study(args)
    runtime = LfmDSparkRuntime(args.aux, args.dspark, model_id=args.model_id, cpu_threads=args.cpu_threads, dtype=args.dtype)
    benchmark_prompts = _read_prompts(args.prompts, args.prompt_limit); calibration_prompts = _read_prompts(args.calibration_prompts, args.calibration_prompt_limit)
    warm = runtime.encode(calibration_prompts[0]); _ = runtime.target_logits(warm); _ = runtime.draft_logits(runtime.context_features(warm))
    selected, calibration = calibrate_v11(runtime, calibration_prompts, tokens=args.calibration_tokens)
    rows, focus_summary, outputs = benchmark_focus(runtime, benchmark_prompts, tokens=args.tokens, repeats=args.repeats, top_k=args.top_k, dspark_floor=float(base["dspark_v9"]["selected_survival_floor"]), v10_config=V10Config(**base["boltzmann_v10"]["selected_config"]), v11_config=selected)
    ordered = sorted(FOCUS_METHODS, key=lambda m: focus_summary[m]["tokens_per_second_median"], reverse=True); winner = ordered[0]; lead = float(focus_summary[ordered[0]]["tokens_per_second_median"] / max(focus_summary[ordered[1]]["tokens_per_second_median"], 1e-12) - 1.0)
    for method in FOCUS_METHODS: base["summary"][method] = focus_summary[method]
    base["summary"][V11_METHOD] = focus_summary[V11_METHOD]; base["runs"].extend(rows[V11_METHOD]); base["winner"] = winner; base["showcase"]["version"] = 11
    base["method_scope"] = "V11 uses deterministic two-way Boltzmann uncertainty only as a gate; only weak DFlash slots pay sparse MOBS dependency correction. Current-contender timings use rotated execution order; exact LFM target verification is authoritative."
    base["v11"] = {"selected_config": config_dict(selected), "calibration": calibration, "training_steps": 0, "focus_methods": list(FOCUS_METHODS), "focus_summary": focus_summary, "focus_winner": winner, "lead_vs_next_fraction": lead, "clear_winner_threshold_fraction": 0.02, "clear_winner": bool(winner == V11_METHOD and lead >= 0.02), "outputs": outputs, "mechanism": "DFlash fast path + Boltzmann uncertainty gate + sparse MOBS correction."}
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True); (out / "benchmark.json").write_text(json.dumps(base, indent=2), encoding="utf-8"); build_report(base, out / "report.html"); return base


def main() -> None:
    p = argparse.ArgumentParser(description="V11 Boltzmann-Gated MOBS LFM CPU study")
    p.add_argument("--aux", default="lfm-artifacts/lfm_aux.pt"); p.add_argument("--dspark", default="lfm-artifacts/lfm_dspark.pt"); p.add_argument("--prompts", default="real_benchmarks/prompts.json"); p.add_argument("--calibration-prompts", default="real_benchmarks/calibration_prompts.json"); p.add_argument("--output-dir", default="lfm-reports"); p.add_argument("--model-id", default="LiquidAI/LFM2.5-350M-Base")
    p.add_argument("--tokens", type=int, default=24); p.add_argument("--repeats", type=int, default=3); p.add_argument("--prompt-limit", type=int, default=6); p.add_argument("--calibration-tokens", type=int, default=8); p.add_argument("--calibration-prompt-limit", type=int, default=4); p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--jump-weight", type=float, default=0.5); p.add_argument("--fused-weight", type=float, default=1.0); p.add_argument("--fused-min-margin", type=float, default=0.0); p.add_argument("--boltzmann-temperature", type=float, default=0.15); p.add_argument("--bmobs-temperature", type=float, default=0.35); p.add_argument("--cpu-threads", type=int, default=2); p.add_argument("--dtype", choices=("float32",), default="float32")
    args = p.parse_args(); d = run(args); print(json.dumps({"winner": d["winner"], "focus_summary": d["v11"]["focus_summary"], "v11": {"selected_config": d["v11"]["selected_config"], "lead_vs_next_fraction": d["v11"]["lead_vs_next_fraction"], "clear_winner": d["v11"]["clear_winner"]}}, indent=2))


if __name__ == "__main__": main()
