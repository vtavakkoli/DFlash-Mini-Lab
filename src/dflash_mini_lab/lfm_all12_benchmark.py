from __future__ import annotations

import argparse
import html
import json
import statistics
from pathlib import Path

import numpy as np

from .lfm_benchmark import _read_prompts, normal_decode, speculative_decode
from .lfm_dspark import LfmDSparkRuntime
from .lfm_showcase import calibrate_act, dflash7_decode
from .lfm_v9v10_benchmark import calibrate_dspark, calibrate_v10, dspark_decode, v10_decode
from .v11_benchmark import V11_METHOD, calibrate_v11, v11_decode
from .v12_benchmark import V12_METHOD, v12_decode
from .v12_parareal import V12Config, load_linear_model


LEGACY_METHODS = (
    "dflash",
    "dflash2",
    "dflash3_mobs",
    "dflash4_jump_mobs",
    "dflash5_fused_jump_mobs",
    "dflash6_boltzmann",
    "dflash6_bmobs",
)

METHODS = (
    "normal",
    *LEGACY_METHODS,
    "dflash7_act",
    "dspark_v9",
    "boltzmann_v10",
    V11_METHOD,
    V12_METHOD,
)

METHOD_INFO = {
    "normal": {"index": 0, "label": "Normal autoregressive", "short": "Target-only greedy baseline", "mechanism": "One authoritative LFM2.5 target step per generated token.", "complexity": "N target steps", "stages": ["TARGET", "TOKEN", "TARGET", "TOKEN"]},
    "dflash": {"index": 1, "label": "DFlash", "short": "Parallel block argmax", "mechanism": "Draft all future block positions in parallel, then verify the complete proposal with LFM2.5.", "complexity": "parallel draft + verify", "stages": ["DRAFT", "BLOCK", "VERIFY", "COMMIT"]},
    "dflash2": {"index": 2, "label": "DFlash2", "short": "Top-k dynamic-programming path", "mechanism": "Retain top-k candidates per slot and choose a predecessor-aware path before exact verification.", "complexity": "O(BK²) guidance", "stages": ["TOP-K", "DP PATH", "VERIFY", "COMMIT"]},
    "dflash3_mobs": {"index": 3, "label": "DFlash3-MOBS", "short": "Middle-out bidirectional selection", "mechanism": "Choose a central anchor and expand left/right using linear-in-K neighbor scoring.", "complexity": "O(BK) guidance", "stages": ["TOP-K", "MIDDLE-OUT", "VERIFY", "COMMIT"]},
    "dflash4_jump_mobs": {"index": 4, "label": "DFlash4-JUMP-MOBS", "short": "Sparse future anchors", "mechanism": "Predict sparse jump anchors, then fill the gaps with MOBS before target verification.", "complexity": "O(BK + JK) + jump pass", "stages": ["JUMP", "GAP FILL", "VERIFY", "COMMIT"]},
    "dflash5_fused_jump_mobs": {"index": 5, "label": "DFlash5-FUSED-JUMP", "short": "Fused sparse guidance", "mechanism": "Reuse drafter hidden states for sparse residual anchors, removing DFlash4's extra jump forward.", "complexity": "O(BK + JKR)", "stages": ["FUSED", "ANCHORS", "VERIFY", "COMMIT"]},
    "dflash6_boltzmann": {"index": 6, "label": "DFlash6-Boltzmann", "short": "Deterministic Boltzmann exploration", "mechanism": "Use confidence-adaptive deterministic Gumbel/Boltzmann scoring over retained candidates.", "complexity": "O(BK) guidance", "stages": ["TOP-K", "BOLTZMANN", "VERIFY", "COMMIT"]},
    "dflash6_bmobs": {"index": 7, "label": "DFlash6-BMOBS", "short": "Boltzmann anchor + MOBS", "mechanism": "Explore one uncertain middle anchor, then fill the remaining block with MOBS.", "complexity": "O(BK) guidance", "stages": ["ANCHOR", "MOBS FILL", "VERIFY", "COMMIT"]},
    "dflash7_act": {"index": 8, "label": "DFlash7-ACT", "short": "Adaptive speculation horizon", "mechanism": "Use draft margins to trim an uncertain suffix before expensive target verification.", "complexity": "O(B) routing", "stages": ["DRAFT", "TRIM", "VERIFY", "COMMIT"]},
    "dspark_v9": {"index": 9, "label": "V9 DSpark-Lite", "short": "Markov correction + survival gate", "mechanism": "Apply a low-rank previous-token correction and learned prefix-survival confidence.", "complexity": "O(BK) guidance", "stages": ["MARKOV", "SURVIVAL", "VERIFY", "COMMIT"]},
    "boltzmann_v10": {"index": 10, "label": "V10 Advanced Boltzmann", "short": "Sparse uncertainty routing", "mechanism": "Keep confident slots on the argmax fast path and spend candidate work only on uncertain positions.", "complexity": "sparse O(BK)", "stages": ["MARGIN", "SPARSE SAMPLE", "VERIFY", "COMMIT"]},
    V11_METHOD: {"index": 11, "label": "V11 Boltzmann-Gated MOBS", "short": "Uncertainty-gated sparse MOBS", "mechanism": "Route only the most uncertain positions through deterministic Boltzmann-gated MOBS correction.", "complexity": "bounded sparse O(BK)", "stages": ["GATE", "SPARSE MOBS", "VERIFY", "COMMIT"]},
    V12_METHOD: {"index": 12, "label": "V12 PARAREAL", "short": "Parallel linear residual correction", "mechanism": "Apply two vectorized affine F-G residual-correction rounds to the whole top-k score field, then verify exactly.", "complexity": "O(RBKD) linear algebra", "stages": ["COARSE G", "F-G ×2", "VERIFY", "COMMIT"]},
}


def _greedy_reference(runtime: LfmDSparkRuntime, ids: np.ndarray, tokens: int) -> np.ndarray:
    seq = np.asarray(ids, dtype=np.int64).copy()
    for _ in range(int(tokens)):
        logits = runtime.target_logits(seq)
        seq = np.append(seq, int(np.argmax(logits[-1])))
    return seq


def _v12_config(model, top_k: int) -> V12Config:
    raw = dict((model.metadata or {}).get("config", {}))
    return V12Config(top_k=int(top_k), correction_rounds=int(raw.get("correction_rounds", 2)), damping=float(raw.get("damping", 0.75)), ridge=float(raw.get("ridge", 1e-3)), residual_clip=float(raw.get("residual_clip", 6.0)), interpolation=tuple(float(x) for x in raw.get("interpolation", (0.0, 0.5, 0.75))))


def _summary(rows: list[dict]) -> dict:
    mean = lambda key, default=0.0: float(statistics.fmean(float(row.get(key, default)) for row in rows))
    median = lambda key, default=0.0: float(statistics.median(float(row.get(key, default)) for row in rows))
    guidance = [float(row.get("total_guidance_scores", 0.0)) + float(row.get("v12_linear_candidate_scores", 0.0)) for row in rows]
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
    for key in ("act_mean_verified_drafts_per_block", "act_trimmed_draft_tokens", "dspark_mean_confidence", "v10_fast_argmax_positions", "v11_fast_argmax_positions", "v12_mean_update_rms"):
        if any(key in row for row in rows): result[key] = mean(key)
    return result


def _run_method(method: str, runtime: LfmDSparkRuntime, ids: np.ndarray, tokens: int, *, top_k: int, jump_weight: float, fused_weight: float, boltzmann_temperature: float, bmobs_temperature: float, act_threshold: float, dspark_floor: float, v10_config, v11_config, v12_model, v12_config: V12Config):
    if method == "normal":
        out, stats = normal_decode(runtime, ids, tokens); return out, stats, {}
    if method in LEGACY_METHODS:
        out, stats = speculative_decode(runtime, ids, tokens, method, top_k=top_k, jump_weight=jump_weight, fused_weight=fused_weight, boltzmann_temperature=boltzmann_temperature, bmobs_temperature=bmobs_temperature); return out, stats, {}
    if method == "dflash7_act": return dflash7_decode(runtime, ids, tokens, margin_threshold=act_threshold)
    if method == "dspark_v9": return dspark_decode(runtime, ids, tokens, top_k=top_k, survival_floor=dspark_floor)
    if method == "boltzmann_v10": return v10_decode(runtime, ids, tokens, config=v10_config)
    if method == V11_METHOD: return v11_decode(runtime, ids, tokens, config=v11_config)
    if method == V12_METHOD: return v12_decode(runtime, ids, tokens, model=v12_model, config=v12_config)
    raise ValueError(method)


def benchmark(runtime: LfmDSparkRuntime, prompts: list[str], *, tokens: int, repeats: int, top_k: int, jump_weight: float, fused_weight: float, boltzmann_temperature: float, bmobs_temperature: float, act_threshold: float, dspark_floor: float, v10_config, v11_config, v12_model, v12_config: V12Config):
    rows = {method: [] for method in METHODS}; examples = []
    for prompt_index, prompt in enumerate(prompts):
        ids = runtime.encode(prompt); reference = _greedy_reference(runtime, ids, tokens); first_text = {}
        for repeat in range(max(1, int(repeats))):
            shift = (prompt_index * max(1, int(repeats)) + repeat) % len(METHODS); order = METHODS[shift:] + METHODS[:shift]
            for method in order:
                output, stats, meta = _run_method(method, runtime, ids, tokens, top_k=top_k, jump_weight=jump_weight, fused_weight=fused_weight, boltzmann_temperature=boltzmann_temperature, bmobs_temperature=bmobs_temperature, act_threshold=act_threshold, dspark_floor=dspark_floor, v10_config=v10_config, v11_config=v11_config, v12_model=v12_model, v12_config=v12_config)
                row = stats.to_dict(); row.update(meta); row.update(prompt=prompt, prompt_index=int(prompt_index), repeat=int(repeat), exact_match=bool(np.array_equal(output, reference))); rows[method].append(row); first_text.setdefault(method, runtime.decode(output))
        examples.append({"prompt": prompt, "reference": runtime.decode(reference), "outputs": first_text})
    summary = {method: _summary(rows[method]) for method in METHODS}; normal_tps = float(summary["normal"]["tokens_per_second_median"])
    for method in METHODS: summary[method]["speedup_vs_normal"] = float(summary[method]["tokens_per_second_median"] / max(normal_tps, 1e-12))
    return rows, summary, examples


def _report_html(data: dict) -> str:
    summary = data["summary"]; methods = list(METHODS); speculative = methods[1:]; winner = max(speculative, key=lambda m: summary[m]["tokens_per_second_median"]); best = summary[winner]; normal = summary["normal"]; all_exact = all(summary[m]["all_exact"] for m in methods)
    payload = json.dumps({"summary": summary, "methods": METHOD_INFO, "order": methods, "winner": winner, "model": data["model"]}, separators=(",", ":")).replace("</", "<\\/")
    table_rows = []
    for method in sorted(methods, key=lambda m: summary[m]["speedup_vs_normal"], reverse=True):
        info, s = METHOD_INFO[method], summary[method]; cls = " winner" if method == winner else (" baseline" if method == "normal" else "")
        table_rows.append(f"<tr class='{cls}'><td><span class='method-no'>{info['index']:02d}</span><b>{html.escape(info['label'])}</b><small>{html.escape(info['short'])}</small></td><td>{s['tokens_per_second_median']:.3f}</td><td><b>{s['speedup_vs_normal']:.3f}×</b></td><td>{100*s['mean_acceptance_rate']:.1f}%</td><td>{s['mean_target_forward_passes']:.2f}</td><td>{s['mean_tokens_per_target_pass']:.2f}</td><td>{s['mean_guidance_work']:.1f}</td><td>{'✓' if s['all_exact'] else '✗'}</td></tr>")
    method_options = "".join(f"<option value='{m}'>{METHOD_INFO[m]['index']:02d} · {html.escape(METHOD_INFO[m]['label'])}</option>" for m in speculative)
    evolution = "".join(f"<button class='evo' data-method='{m}'><span>{METHOD_INFO[m]['index']:02d}</span><b>{html.escape(METHOD_INFO[m]['label'])}</b><small>{html.escape(METHOD_INFO[m]['short'])}</small></button>" for m in speculative)
    model = data["model"]
    css = ":root{font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;color:#eaf3ff;background:#06111d;--panel:#0c1b2a;--line:#24425c;--muted:#90a9be;--cyan:#5bdcff;--green:#82f0b2;--amber:#ffd479}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:radial-gradient(circle at 12% -5%,#164d73 0,transparent 34%),radial-gradient(circle at 92% 12%,#17394e 0,transparent 28%),#06111d;color:#eaf3ff}a{color:var(--cyan)}.wrap{max-width:1320px;margin:auto;padding:28px 22px 80px}.topbar{display:flex;justify-content:space-between;align-items:center;gap:14px;margin-bottom:28px}.brand{font-weight:800}.badge{border:1px solid #31536e;border-radius:999px;padding:7px 11px;color:#c8dae8;background:#0c1d2d}.hero{display:grid;grid-template-columns:1.5fr .8fr;gap:18px}.panel{background:linear-gradient(145deg,#10283d,#091725);border:1px solid var(--line);border-radius:22px;box-shadow:0 22px 60px #0004}.hero-main{padding:34px}.eyebrow{text-transform:uppercase;letter-spacing:.16em;color:var(--cyan);font-size:.78rem;font-weight:800}.hero h1{font-size:clamp(2.5rem,6vw,5.2rem);line-height:.92;margin:14px 0 18px;letter-spacing:-.055em}.hero p{color:#bed1e0;font-size:1.07rem;line-height:1.65}.hero-side{padding:26px;display:grid;align-content:center;gap:12px}.hero-stat{border:1px solid #294966;background:#081827;border-radius:16px;padding:16px}.hero-stat b{display:block;font-size:1.55rem;color:var(--green)}.hero-stat span{color:var(--muted);font-size:.86rem}.kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:18px 0 28px}.kpi{padding:18px}.kpi strong{display:block;font-size:1.55rem}.kpi small{color:var(--muted)}.section{margin-top:26px;padding:26px}.section-head{display:flex;justify-content:space-between;gap:18px;align-items:end;margin-bottom:18px}.section h2{margin:0;font-size:1.55rem}.section-head p{margin:0;color:var(--muted);max-width:700px}.chart{display:grid;gap:8px}.bar-row{display:grid;grid-template-columns:210px 1fr 82px;gap:10px;align-items:center}.bar-label{font-size:.86rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.track{height:26px;border:1px solid #233f57;border-radius:8px;background:#071521;position:relative;overflow:hidden}.track:after{content:'1×';position:absolute;left:78%;top:2px;bottom:2px;border-left:1px dashed #ffffff55;color:#ffffff88;font-size:.65rem;padding-left:4px}.fill{height:100%;border-radius:7px;background:linear-gradient(90deg,#2e789d,#62e3ff);min-width:2px}.fill.fast{background:linear-gradient(90deg,#2c8a61,#82f0b2)}.bar-value{text-align:right;font-weight:700}.grid2{display:grid;grid-template-columns:1.1fr .9fr;gap:18px}.scatter-wrap{min-height:410px}.scatter{width:100%;height:360px;background:#071521;border:1px solid #203d56;border-radius:16px}.axis-label{fill:#8fa8bd;font-size:12px}.dot-label{fill:#dcecff;font-size:10px}.table-wrap{overflow:auto;border:1px solid #203d56;border-radius:16px}table{width:100%;border-collapse:collapse;min-width:980px}th,td{padding:12px 13px;border-bottom:1px solid #1b354a;text-align:right}thead th{background:#0b1c2b;color:#9eb4c7;font-size:.76rem;text-transform:uppercase}th:first-child,td:first-child{text-align:left}td:first-child b,td:first-child small{display:block}td:first-child small{color:var(--muted);margin-top:3px}.method-no{display:inline-grid;place-items:center;width:28px;height:28px;border:1px solid #31536e;border-radius:8px;margin-right:9px;color:var(--cyan);font-size:.74rem}.winner{background:#0e3128}.baseline{background:#101d2b}.evolution{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.evo{appearance:none;text-align:left;color:inherit;background:#081827;border:1px solid #27465f;border-radius:14px;padding:14px;cursor:pointer}.evo:hover,.evo.active{border-color:var(--cyan);box-shadow:0 0 0 1px #5bdcff33}.evo span{display:inline-grid;place-items:center;width:28px;height:28px;border-radius:8px;background:#12354c;color:var(--cyan);font-weight:800}.evo b,.evo small{display:block;margin-top:8px}.evo small{color:var(--muted)}.explorer{display:grid;grid-template-columns:.8fr 1.2fr;gap:18px}.control,.sim{padding:18px;border:1px solid #24425c;border-radius:16px;background:#081827}select{width:100%;background:#0c2234;border:1px solid #31536e;color:#eaf3ff;border-radius:11px;padding:11px}.method-title{font-size:1.55rem;margin:18px 0 4px}.metric-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:15px}.metric{padding:12px;border:1px solid #203d56;border-radius:11px}.metric b{display:block}.metric span,.mechanism,.sim-note,.foot{color:var(--muted)}.sim{background:#071521;min-height:360px}.sim-head{display:flex;justify-content:space-between}.sim-status{color:var(--green);font-weight:700}.pipeline{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:26px 0}.stage{min-height:74px;border:1px solid #28475f;border-radius:12px;display:grid;place-items:center;color:#8ea6b9}.stage.active{border-color:var(--cyan);color:#fff;background:#0f3047}.tokens{display:flex;gap:7px;flex-wrap:wrap}.token{width:46px;height:42px;border:1px solid #31536e;border-radius:9px;display:grid;place-items:center;background:#0b2030}.token.draft{border-style:dashed}.token.accept{background:#103b2c;border-color:#4ec989}.token.reject{background:#3b1f26;border-color:#b55e70}.token.trim{opacity:.25}.exact{color:var(--green)}.foot{font-size:.83rem;line-height:1.6;margin-top:22px}@media(max-width:980px){.hero,.grid2,.explorer{grid-template-columns:1fr}.kpis{grid-template-columns:1fr 1fr}.evolution{grid-template-columns:1fr 1fr}.bar-row{grid-template-columns:145px 1fr 64px}}@media(max-width:620px){.kpis{grid-template-columns:1fr}.evolution{grid-template-columns:1fr}.pipeline{grid-template-columns:1fr 1fr}}"
    js = """const DATA=JSON.parse(document.getElementById('payload').textContent),S=DATA.summary,M=DATA.methods,O=DATA.order;const f=(x,n=3)=>Number(x).toFixed(n);function bars(){let r=document.getElementById('bars'),mx=Math.max(...O.map(k=>S[k].speedup_vs_normal),1.15);r.innerHTML=O.map(k=>`<div class='bar-row'><div class='bar-label'>${String(M[k].index).padStart(2,'0')} · ${M[k].label}</div><div class='track'><div class='fill ${S[k].speedup_vs_normal>=1?'fast':''}' style='width:${Math.max(1,100*S[k].speedup_vs_normal/mx)}%'></div></div><div class='bar-value'>${f(S[k].speedup_vs_normal)}×</div></div>`).join('')}function scatter(){let q=O.filter(k=>k!='normal'),svg=document.getElementById('scatter'),W=720,H=350,p=48,xm=Math.max(.12,...q.map(k=>S[k].mean_acceptance_rate))*1.08,yn=Math.min(.95,...q.map(k=>S[k].speedup_vs_normal))*.98,yx=Math.max(1.05,...q.map(k=>S[k].speedup_vs_normal))*1.02,x=v=>p+(W-2*p)*v/xm,y=v=>H-p-(H-2*p)*(v-yn)/(yx-yn),o=`<line x1='${p}' y1='${H-p}' x2='${W-p}' y2='${H-p}' stroke='#45647b'/><line x1='${p}' y1='${p}' x2='${p}' y2='${H-p}' stroke='#45647b'/><line x1='${p}' y1='${y(1)}' x2='${W-p}' y2='${y(1)}' stroke='#82f0b266' stroke-dasharray='5 5'/><text x='${W/2}' y='${H-8}' text-anchor='middle' class='axis-label'>Draft acceptance rate →</text>`;for(let k of q){let cx=x(S[k].mean_acceptance_rate),cy=y(S[k].speedup_vs_normal);o+=`<circle cx='${cx}' cy='${cy}' r='${k==DATA.winner?8:6}' fill='${k==DATA.winner?'#82f0b2':'#5bdcff'}'/><text x='${cx+9}' y='${cy+4}' class='dot-label'>${M[k].index}</text>`}svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.innerHTML=o}let timer=null,cycle=0;function choose(k){document.getElementById('methodSelect').value=k;document.querySelectorAll('.evo').forEach(e=>e.classList.toggle('active',e.dataset.method==k));let m=M[k],s=S[k];methodTitle.textContent=m.label;methodShort.textContent=m.short;mechanism.textContent=m.mechanism;complexity.textContent=m.complexity;metricGrid.innerHTML=`<div class='metric'><b>${f(s.tokens_per_second_median)}</b><span>tokens / second</span></div><div class='metric'><b>${f(s.speedup_vs_normal)}×</b><span>vs normal</span></div><div class='metric'><b>${(100*s.mean_acceptance_rate).toFixed(1)}%</b><span>draft acceptance</span></div><div class='metric'><b>${s.mean_target_forward_passes.toFixed(2)}</b><span>target calls</span></div><div class='metric'><b>${s.mean_tokens_per_target_pass.toFixed(2)}</b><span>tokens / target call</span></div><div class='metric'><b>${s.mean_guidance_work.toFixed(1)}</b><span>guidance work</span></div>`;cycle=0;sim(k,0);if(timer)clearInterval(timer);let ph=0;timer=setInterval(()=>{ph=(ph+1)%4;if(!ph)cycle++;sim(k,ph)},900)}function sim(k,ph){let m=M[k],s=S[k],st=m.stages;simStatus.textContent=st[ph];pipeline.innerHTML=st.map((x,i)=>`<div class='stage ${i==ph?'active':''}'>${x}</div>`).join('');let n=4,a=0,g=((cycle*37+m.index*17)%100)/100;if(k!='normal'&&g<s.mean_acceptance_rate)a=1;if(k!='normal'&&g<s.mean_acceptance_rate*.22)a=2;let vis=k=='dflash7_act'?Math.max(1,Math.min(n,Math.round(s.act_mean_verified_drafts_per_block||n))):n,t=[];for(let i=0;i<n;i++){let c='draft';if(i>=vis)c+=' trim';if(ph>=2&&i<vis)c=i<a?'accept':i==a?'reject':'draft';if(k=='normal')c=ph>=1&&i==0?'accept':'draft';t.push(`<div class='token ${c}'>t+${i+1}</div>`)}tokens.innerHTML=t.join('');simNote.textContent=k=='normal'?'Sequential target-only decoding: one authoritative token at a time.':`${m.label}: mechanism animation using aggregate measured acceptance ${(100*s.mean_acceptance_rate).toFixed(1)}%. The animation is illustrative; benchmark values are measured.`}methodSelect.onchange=e=>choose(e.target.value);document.querySelectorAll('.evo').forEach(e=>e.onclick=()=>{choose(e.dataset.method);explorer.scrollIntoView({behavior:'smooth'})});bars();scatter();choose(DATA.winner);"""
    exact_text = "All 13 paths exact" if all_exact else "Exactness failure detected"
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><meta name='color-scheme' content='dark'><title>DFlash Mini Lab · LFM2.5 All-12 Study</title><style>{css}</style></head><body><main class='wrap'><div class='topbar'><div class='brand'>DFlash Mini Lab</div><div class='badge'>Canonical benchmark · LFM2.5-350M only</div></div><section class='hero'><div class='panel hero-main'><div class='eyebrow'>One model · one CPU protocol · twelve speculative methods</div><h1>LFM2.5<br>All-12 Study</h1><p>A consolidated, exactness-gated comparison of twelve speculative-decoding mechanisms against the same <b>LiquidAI/LFM2.5-350M-Base</b> target. Cross-model Qwen/EAGLE results are intentionally excluded from this site.</p></div><aside class='panel hero-side'><div class='hero-stat'><b>{best['speedup_vs_normal']:.3f}×</b><span>best speculative speedup vs normal</span></div><div class='hero-stat'><b>{html.escape(METHOD_INFO[winner]['label'])}</b><span>fastest speculative method in this run</span></div><div class='hero-stat'><b class='exact'>{'✓' if all_exact else '✗'} {exact_text}</b><span>final output compared with normal greedy LFM</span></div></aside></section><section class='kpis'><div class='panel kpi'><strong>{normal['tokens_per_second_median']:.3f}</strong><small>normal tok/s</small></div><div class='panel kpi'><strong>{best['tokens_per_second_median']:.3f}</strong><small>best speculative tok/s</small></div><div class='panel kpi'><strong>12</strong><small>speculative methods</small></div><div class='panel kpi'><strong>{int(model['target_parameter_count'])/1e6:.1f}M</strong><small>target parameters</small></div><div class='panel kpi'><strong>{int(model['candidate_size']):,}</strong><small>retained draft vocabulary</small></div></section><section class='panel section'><div class='section-head'><div><div class='eyebrow'>Measured performance</div><h2>Speedup versus normal decoding</h2></div><p>Bars use median generated tokens/second from the same LFM2.5 CPU run. The dashed marker is 1× normal.</p></div><div id='bars' class='chart'></div></section><section class='grid2'><section class='panel section scatter-wrap'><div class='section-head'><div><div class='eyebrow'>Efficiency frontier</div><h2>Acceptance × speed</h2></div></div><svg id='scatter' class='scatter'></svg></section><section class='panel section'><div class='section-head'><div><div class='eyebrow'>Evolution</div><h2>12 mechanisms</h2></div></div><div class='evolution'>{evolution}</div></section></section><section class='panel section'><div class='section-head'><div><div class='eyebrow'>Exactness-gated results</div><h2>Complete benchmark table</h2></div><p>Normal is the target-only reference. Every speculative output must match it exactly.</p></div><div class='table-wrap'><table><thead><tr><th>Method</th><th>tok/s</th><th>vs normal</th><th>accept</th><th>target calls</th><th>tok/call</th><th>guidance</th><th>exact</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div></section><section class='panel section' id='explorer'><div class='section-head'><div><div class='eyebrow'>Method explorer</div><h2>Mechanism simulation</h2></div><p>The animation explains each decoder using its measured aggregate profile; it is not a substitute for the benchmark numbers.</p></div><div class='explorer'><div class='control'><label for='methodSelect'>Speculative method</label><select id='methodSelect'>{method_options}</select><h3 class='method-title' id='methodTitle'></h3><div id='methodShort' class='eyebrow'></div><p id='mechanism' class='mechanism'></p><p><b>Guidance complexity:</b> <span id='complexity'></span></p><div id='metricGrid' class='metric-grid'></div></div><div class='sim'><div class='sim-head'><b>Animated decoding block</b><span id='simStatus' class='sim-status'></span></div><div id='pipeline' class='pipeline'></div><div id='tokens' class='tokens'></div><p id='simNote' class='sim-note'></p></div></div></section><section class='panel section'><div class='section-head'><div><div class='eyebrow'>Protocol</div><h2>What this page measures</h2></div></div><div class='grid2'><div><p><b>Target:</b> {html.escape(model['id'])}</p><p><b>Workload:</b> {data['config']['prompt_count']} held-out prompts × {data['config']['max_new_tokens']} generated tokens × {data['config']['repeats']} repeats.</p><p><b>CPU:</b> {data['config']['cpu_threads']} threads · float32.</p></div><div><p><b>Calibration:</b> ACT, V9, V10 and V11 use separate calibration prompts. V12 regression uses training trajectories only.</p><p><b>Exactness:</b> <span class='exact'>target verification is authoritative</span>.</p><p><b>Data:</b> <a href='./benchmark.json'>benchmark.json</a></p></div></div></section><p class='foot'>DFlash Mini Lab is a mechanism-level research/reference repository. DFlash3–DFlash12 are experimental lab variants, not upstream official DFlash releases. Compare throughput only within this matched run.</p><script id='payload' type='application/json'>{payload}</script><script>{js}</script></main></body></html>"""


def run(args: argparse.Namespace) -> dict:
    runtime = LfmDSparkRuntime(args.aux, args.dspark, cpu_threads=args.cpu_threads, dtype=args.dtype)
    prompts = _read_prompts(args.prompts, args.prompt_limit); calibration_prompts = _read_prompts(args.calibration_prompts, args.calibration_prompt_limit)
    if not prompts or not calibration_prompts: raise ValueError("benchmark and calibration prompt sets must be non-empty")
    warm_ids = runtime.encode(prompts[0]); _ = runtime.target_logits(warm_ids); _ = runtime.draft_logits(runtime.context_features(warm_ids))
    act_threshold, act_cal = calibrate_act(runtime, calibration_prompts, tokens=args.calibration_tokens)
    dspark_floor, dspark_cal = calibrate_dspark(runtime, calibration_prompts, tokens=args.calibration_tokens, top_k=args.top_k)
    v10_config, v10_cal = calibrate_v10(runtime, calibration_prompts, tokens=args.calibration_tokens)
    v11_config, v11_cal = calibrate_v11(runtime, calibration_prompts, tokens=args.calibration_tokens)
    v12_model = load_linear_model(args.v12_model); v12_config = _v12_config(v12_model, args.top_k)
    rows, summary, examples = benchmark(runtime, prompts, tokens=args.tokens, repeats=args.repeats, top_k=args.top_k, jump_weight=args.jump_weight, fused_weight=args.fused_weight, boltzmann_temperature=args.boltzmann_temperature, bmobs_temperature=args.bmobs_temperature, act_threshold=act_threshold, dspark_floor=dspark_floor, v10_config=v10_config, v11_config=v11_config, v12_model=v12_model, v12_config=v12_config)
    winner = max(METHODS[1:], key=lambda m: summary[m]["tokens_per_second_median"])
    data = {"study": "LFM2.5 All-12 speculative decoding study", "model": {"id": runtime.model_id, "target_parameter_count": int(runtime.target_parameter_count), "target_vocab_size": int(runtime.target.config.vocab_size), "candidate_size": int(runtime.candidate_size), "aux_parameter_count": int(runtime.aux_parameter_count)}, "config": {"max_new_tokens": int(args.tokens), "repeats": int(args.repeats), "prompt_count": len(prompts), "calibration_prompt_count": len(calibration_prompts), "calibration_tokens": int(args.calibration_tokens), "top_k": int(args.top_k), "cpu_threads": int(args.cpu_threads), "dtype": args.dtype}, "selected_settings": {"act_margin_threshold": float(act_threshold), "dspark_survival_floor": float(dspark_floor), "v10": getattr(v10_config, "__dict__", {}), "v11": getattr(v11_config, "__dict__", {}), "v12": {"top_k": v12_config.top_k, "correction_rounds": v12_config.correction_rounds, "damping": v12_config.damping}}, "calibration": {"dflash7_act": act_cal, "dspark_v9": dspark_cal, "boltzmann_v10": v10_cal, V11_METHOD: v11_cal, V12_METHOD: (v12_model.metadata or {}).get("holdout_convergence", {})}, "method_info": METHOD_INFO, "summary": summary, "rows": rows, "examples": examples, "winner": winner}
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True); (out / "benchmark.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"); report = _report_html(data); (out / "report.html").write_text(report, encoding="utf-8"); (out / "index.html").write_text(report, encoding="utf-8"); return data


def main() -> None:
    p = argparse.ArgumentParser(description="Run the canonical Normal + 12-method LFM2.5 CPU benchmark")
    p.add_argument("--aux", default="lfm-artifacts/lfm_aux.pt"); p.add_argument("--dspark", default="lfm-artifacts/lfm_dspark.pt"); p.add_argument("--v12-model", default="lfm-artifacts/v12_parareal.json"); p.add_argument("--prompts", default="real_benchmarks/prompts.json"); p.add_argument("--calibration-prompts", default="real_benchmarks/calibration_prompts.json"); p.add_argument("--output-dir", default="lfm-reports"); p.add_argument("--tokens", type=int, default=24); p.add_argument("--repeats", type=int, default=2); p.add_argument("--prompt-limit", type=int, default=6); p.add_argument("--calibration-tokens", type=int, default=8); p.add_argument("--calibration-prompt-limit", type=int, default=4); p.add_argument("--top-k", type=int, default=8); p.add_argument("--jump-weight", type=float, default=0.5); p.add_argument("--fused-weight", type=float, default=1.0); p.add_argument("--boltzmann-temperature", type=float, default=0.15); p.add_argument("--bmobs-temperature", type=float, default=0.35); p.add_argument("--cpu-threads", type=int, default=2); p.add_argument("--dtype", default="float32", choices=("float32", "bfloat16")); args = p.parse_args(); data = run(args); print(json.dumps({"winner": data["winner"], "model": data["model"]["id"], "summary": data["summary"]}, indent=2))


if __name__ == "__main__": main()
