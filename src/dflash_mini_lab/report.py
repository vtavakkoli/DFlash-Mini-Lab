from __future__ import annotations

import base64
import html
import json
from pathlib import Path


def _b64(path: str | Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _fmt(v: float, digits: int = 2) -> str:
    return f"{float(v):.{digits}f}"


def build_report(benchmark_json: str | Path, speed_gif: str | Path, architecture_gif: str | Path, out_html: str | Path) -> None:
    data = json.loads(Path(benchmark_json).read_text(encoding="utf-8"))
    summary = data["summary"]
    methods = ("normal", "dflash", "dflash2", "dflash3_mobs", "dflash4_jump_mobs")
    labels = {"normal": "Normal", "dflash": "DFlash", "dflash2": "DFlash v2", "dflash3_mobs": "DFlash3-MOBS", "dflash4_jump_mobs": "DFlash4-JUMP-MOBS"}
    rows = []
    for key in methods:
        s = summary[key]
        guidance = "—" if not s["mean_total_guidance_scores"] else _fmt(s["mean_total_guidance_scores"], 0)
        jump_passes = "—" if not s["mean_jump_forward_passes"] else _fmt(s["mean_jump_forward_passes"], 1)
        rows.append(
            f"<tr><th>{labels[key]}</th><td>{_fmt(s['tokens_per_second_median'])}</td>"
            f"<td>{_fmt(s['latency_ms_median'])}</td><td>{_fmt(s['speedup_vs_normal'])}×</td>"
            f"<td>{_fmt(s['mean_tokens_per_target_pass'])}</td><td>{_fmt(100*s['mean_acceptance_rate'],1)}%</td>"
            f"<td>{guidance}</td><td>{jump_passes}</td><td>{'✓' if s['all_exact'] else '✗'}</td></tr>"
        )
    chart_data = json.dumps({key: {"label": labels[key], "tps": summary[key]["tokens_per_second_median"], "latency": summary[key]["latency_ms_median"], "speedup": summary[key]["speedup_vs_normal"], "targetPass": summary[key]["mean_tokens_per_target_pass"], "guidance": summary[key]["mean_total_guidance_scores"]} for key in methods})

    d4 = summary["dflash4_jump_mobs"]
    work_reduction = 100.0 * d4.get("guidance_work_reduction_vs_dflash2", 0.0)
    d2_delta = 100.0 * (d4.get("throughput_ratio_vs_dflash2", 1.0) - 1.0)
    mobs_delta = 100.0 * (d4.get("throughput_ratio_vs_mobs", 1.0) - 1.0)
    verdict = (
        f"JUMP-MOBS uses {work_reduction:.1f}% fewer measured guidance scores than DFlash2 and is "
        f"{abs(d2_delta):.1f}% {'faster' if d2_delta >= 0 else 'slower'} than DFlash2, and "
        f"{abs(mobs_delta):.1f}% {'faster' if mobs_delta >= 0 else 'slower'} than pure MOBS on this run."
    )
    offsets = ", ".join(f"+{x}" for x in data["config"].get("dflash4_jump_offsets", []))
    jump_weight = float(data["config"].get("dflash4_jump_weight", 0.5))

    css = ":root{font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;color:#182030;background:#f6f8fb}body{margin:0}.wrap{max-width:1280px;margin:auto;padding:34px}.hero,.card{background:white;border:1px solid #d9e0eb;border-radius:16px;box-shadow:0 4px 20px #1b2a4410}.hero{padding:30px;margin-bottom:20px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.card{padding:22px;margin-bottom:20px}.wide{grid-column:1/-1}h1{margin:0 0 8px;font-size:34px}h2{margin:0 0 16px;font-size:22px}p{line-height:1.55;color:#4a586f}.badge{display:inline-block;padding:6px 10px;border-radius:999px;background:#edf3ff;color:#2d64bd;font-weight:700;margin-right:8px}.warning{background:#fff6e8;border-left:4px solid #e58a36;padding:14px 16px;border-radius:8px;color:#65451f}.verdict{background:#eafcff;border-left:4px solid #18a7b8;padding:14px 16px;border-radius:8px;color:#164d55;margin-top:14px;font-weight:650}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:11px;border-bottom:1px solid #e4e9f1;text-align:right}th:first-child,td:first-child{text-align:left}thead th{background:#f5f7fb}.gif{width:100%;border-radius:12px;border:1px solid #e1e6ee}.controls button{border:1px solid #cfd7e5;background:#fff;border-radius:9px;padding:9px 12px;margin:0 6px 10px 0;cursor:pointer}.controls button.active{background:#182030;color:white}.barrow{display:grid;grid-template-columns:165px 1fr 100px;gap:12px;align-items:center;margin:14px 0}.track{height:27px;background:#eef2f7;border-radius:8px;overflow:hidden}.bar{height:100%;border-radius:8px;transition:width .35s ease}.normal{background:#366fd2}.dflash{background:#eb8437}.dflash2{background:#379d64}.dflash3_mobs{background:#7e57c2}.dflash4_jump_mobs{background:#18a7b8}.code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#101827;color:#e9effa;padding:14px;border-radius:10px;overflow:auto}@media(max-width:800px){.grid{grid-template-columns:1fr}.wrap{padding:18px}.wide{grid-column:auto}table{display:block;overflow-x:auto}}"
    js = f"const DATA={chart_data};const meta={{tps:['Tokens / sec',' tok/s'],latency:['Median latency',' ms'],speedup:['Speedup vs normal','×'],targetPass:['Tokens / target pass',''],guidance:['Guidance candidate scores','']}};function render(m){{document.querySelectorAll('.controls button').forEach(b=>b.classList.toggle('active',b.dataset.metric===m));const vals=Object.values(DATA).map(x=>x[m]);const max=Math.max(...vals,1);document.getElementById('metricTitle').textContent=meta[m][0];document.getElementById('bars').innerHTML=Object.entries(DATA).map(([k,x])=>`<div class=\"barrow\"><b>${{x.label}}</b><div class=\"track\"><div class=\"bar ${{k}}\" style=\"width:${{x[m]===0?0:Math.max(4,x[m]/max*100)}}%\"></div></div><span>${{x[m].toFixed(2)}}${{meta[m][1]}}</span></div>`).join('');}}window.addEventListener('DOMContentLoaded',()=>render('tps'));"
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DFlash Mini Lab CPU Benchmark</title><style>{css}</style></head><body><main class="wrap"><section class="hero"><span class="badge">CPU only</span><span class="badge">CI verified</span><span class="badge">Exact target verification</span><h1>DFlash Mini Lab benchmark report</h1><p>Five-way comparison: normal autoregressive decoding, DFlash, DFlash v2, DFlash3-MOBS and experimental <b>DFlash4-JUMP-MOBS</b>.</p><div class="warning"><b>Interpretation:</b> {html.escape(data['timing_note'])}</div><div class="verdict"><b>JUMP result:</b> {html.escape(verdict)}</div></section><section class="card wide"><h2>Performance matrix</h2><table><thead><tr><th>Mode</th><th>Median tok/s</th><th>Median latency</th><th>Speedup</th><th>Tokens / target pass</th><th>Draft acceptance</th><th>Guidance scores</th><th>Jump passes</th><th>Exact</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section><div class="grid"><section class="card"><h2 id="metricTitle">Interactive comparison</h2><div class="controls"><button class="active" data-metric="tps" onclick="render('tps')">Throughput</button><button data-metric="latency" onclick="render('latency')">Latency</button><button data-metric="speedup" onclick="render('speedup')">Speedup</button><button data-metric="targetPass" onclick="render('targetPass')">Target-pass efficiency</button><button data-metric="guidance" onclick="render('guidance')">Guidance work</button></div><div id="bars"></div></section><section class="card"><h2>Mechanism differences</h2><p><b>Normal:</b> one target decision per generated token.</p><p><b>DFlash:</b> predicts a block in parallel, then verifies it.</p><p><b>DFlash v2:</b> top-k candidate lattice with O(BK²) predecessor-conditioned dynamic programming.</p><p><b>DFlash3-MOBS:</b> middle-out O(BK) local path selection.</p><p><b>DFlash4-JUMP-MOBS:</b> a separate tiny indexed head predicts sparse future anchors at {html.escape(offsets)}. Those anchors guide an O(BK) gap-filling selector before exact target verification. Jump weight: {jump_weight:.2f}.</p><p><b>Fidelity:</b> {html.escape(data['fidelity_note'])}</p></section><section class="card"><h2>Speed animation</h2><img class="gif" alt="Animated five-way speed comparison" src="data:image/gif;base64,{_b64(speed_gif)}"></section><section class="card"><h2>JUMP-MOBS architecture animation</h2><img class="gif" alt="Animated DFlash4 JUMP-MOBS architecture" src="data:image/gif;base64,{_b64(architecture_gif)}"></section></div><section class="card wide"><h2>Reproduce</h2><div class="code">docker build -t dflash-mini-lab .\ndocker run --rm -e CPU_THREADS=1 -v \"$PWD/reports:/app/reports\" dflash-mini-lab --top-k {data['config']['dflash4_jump_mobs_top_k']} --jump-weight {jump_weight}</div><p>Workload: {data['config']['prompt_count']} prompts × {data['config']['repeats']} measured repeats × {data['config']['max_new_tokens']} generated tokens; block size {data['config']['dflash_block_size']}; jump offsets {html.escape(offsets)}.</p></section></main><script>{js}</script></body></html>"""
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(document, encoding="utf-8")
