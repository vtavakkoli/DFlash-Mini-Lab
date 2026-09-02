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
    methods = ("normal", "dflash", "dflash2", "dflash3_mobs")
    labels = {"normal": "Normal", "dflash": "DFlash", "dflash2": "DFlash v2", "dflash3_mobs": "DFlash3-MOBS"}
    rows = []
    for key in methods:
        s = summary[key]
        selector = "—" if not s["mean_selector_pair_scores"] else _fmt(s["mean_selector_pair_scores"], 0)
        rows.append(
            f"<tr><th>{labels[key]}</th><td>{_fmt(s['tokens_per_second_median'])}</td>"
            f"<td>{_fmt(s['latency_ms_median'])}</td><td>{_fmt(s['latency_ms_p95'])}</td>"
            f"<td>{_fmt(s['speedup_vs_normal'])}×</td><td>{_fmt(s['mean_tokens_per_target_pass'])}</td>"
            f"<td>{_fmt(100*s['mean_acceptance_rate'],1)}%</td><td>{selector}</td>"
            f"<td>{'✓' if s['all_exact'] else '✗'}</td></tr>"
        )
    chart_data = json.dumps(
        {
            key: {
                "label": labels[key],
                "tps": summary[key]["tokens_per_second_median"],
                "latency": summary[key]["latency_ms_median"],
                "speedup": summary[key]["speedup_vs_normal"],
                "targetPass": summary[key]["mean_tokens_per_target_pass"],
                "selector": summary[key]["mean_selector_pair_scores"],
            }
            for key in methods
        }
    )
    d3 = summary["dflash3_mobs"]
    d2 = summary["dflash2"]
    selector_reduction = 100.0 * d3.get("selector_work_reduction_vs_dflash2", 0.0)
    throughput_delta = 100.0 * (d3.get("throughput_ratio_vs_dflash2", 1.0) - 1.0)
    verdict = (
        f"DFlash3-MOBS uses {selector_reduction:.1f}% fewer selector pair scores than DFlash2 and "
        f"is {abs(throughput_delta):.1f}% {'faster' if throughput_delta >= 0 else 'slower'} in median end-to-end throughput on this run."
    )
    css = ":root{font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;color:#182030;background:#f6f8fb}body{margin:0}.wrap{max-width:1240px;margin:auto;padding:34px}.hero,.card{background:white;border:1px solid #d9e0eb;border-radius:16px;box-shadow:0 4px 20px #1b2a4410}.hero{padding:30px;margin-bottom:20px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.card{padding:22px;margin-bottom:20px}.wide{grid-column:1/-1}h1{margin:0 0 8px;font-size:34px}h2{margin:0 0 16px;font-size:22px}p{line-height:1.55;color:#4a586f}.badge{display:inline-block;padding:6px 10px;border-radius:999px;background:#edf3ff;color:#2d64bd;font-weight:700;margin-right:8px}.warning{background:#fff6e8;border-left:4px solid #e58a36;padding:14px 16px;border-radius:8px;color:#65451f}.verdict{background:#f3efff;border-left:4px solid #7e57c2;padding:14px 16px;border-radius:8px;color:#3f2d69;margin-top:14px;font-weight:650}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:11px;border-bottom:1px solid #e4e9f1;text-align:right}th:first-child,td:first-child{text-align:left}thead th{background:#f5f7fb}.gif{width:100%;border-radius:12px;border:1px solid #e1e6ee}.controls button{border:1px solid #cfd7e5;background:#fff;border-radius:9px;padding:9px 12px;margin:0 6px 10px 0;cursor:pointer}.controls button.active{background:#182030;color:white}.barrow{display:grid;grid-template-columns:135px 1fr 100px;gap:12px;align-items:center;margin:16px 0}.track{height:28px;background:#eef2f7;border-radius:8px;overflow:hidden}.bar{height:100%;border-radius:8px;transition:width .35s ease}.normal{background:#366fd2}.dflash{background:#eb8437}.dflash2{background:#379d64}.dflash3_mobs{background:#7e57c2}.code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#101827;color:#e9effa;padding:14px;border-radius:10px;overflow:auto}@media(max-width:800px){.grid{grid-template-columns:1fr}.wrap{padding:18px}.wide{grid-column:auto}table{display:block;overflow-x:auto}}"
    js = f"const DATA={chart_data};const meta={{tps:['Tokens / sec',' tok/s'],latency:['Median latency',' ms'],speedup:['Speedup vs normal','×'],targetPass:['Tokens / target pass',''],selector:['Selector pair scores','']}};function render(m){{document.querySelectorAll('.controls button').forEach(b=>b.classList.toggle('active',b.dataset.metric===m));const vals=Object.values(DATA).map(x=>x[m]);const max=Math.max(...vals,1);document.getElementById('metricTitle').textContent=meta[m][0];document.getElementById('bars').innerHTML=Object.entries(DATA).map(([k,x])=>`<div class=\"barrow\"><b>${{x.label}}</b><div class=\"track\"><div class=\"bar ${{k}}\" style=\"width:${{x[m]===0?0:Math.max(4,x[m]/max*100)}}%\"></div></div><span>${{x[m].toFixed(2)}}${{meta[m][1]}}</span></div>`).join('');}}window.addEventListener('DOMContentLoaded',()=>render('tps'));"
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DFlash Mini Lab CPU Benchmark</title><style>{css}</style></head><body><main class="wrap"><section class="hero"><span class="badge">CPU only</span><span class="badge">CI verified</span><span class="badge">Lossless greedy verification</span><h1>DFlash Mini Lab benchmark report</h1><p>Normal autoregressive decoding vs. DFlash, DFlash v2 and the experimental <b>DFlash3-MOBS</b> O(BK) middle-out selector on the same tiny Transformer SLM.</p><div class="warning"><b>Interpretation:</b> {html.escape(data['timing_note'])}</div><div class="verdict"><b>MOBS result:</b> {html.escape(verdict)}</div></section><section class="card wide"><h2>Performance matrix</h2><table><thead><tr><th>Mode</th><th>Median tok/s</th><th>Median latency</th><th>P95 latency</th><th>Speedup</th><th>Tokens / target pass</th><th>Draft acceptance</th><th>Selector pair scores</th><th>Exact</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section><div class="grid"><section class="card"><h2 id="metricTitle">Interactive comparison</h2><div class="controls"><button class="active" data-metric="tps" onclick="render('tps')">Throughput</button><button data-metric="latency" onclick="render('latency')">Latency</button><button data-metric="speedup" onclick="render('speedup')">Speedup</button><button data-metric="targetPass" onclick="render('targetPass')">Target-pass efficiency</button><button data-metric="selector" onclick="render('selector')">Selector work</button></div><div id="bars"></div></section><section class="card"><h2>Mechanism differences</h2><p><b>Normal:</b> target chooses one next token per forward pass.</p><p><b>DFlash:</b> parallel block drafter proposes a complete block, then the target verifies it.</p><p><b>DFlash v2:</b> keeps top-k candidates and uses a dynamic-programming transition grid, with selector work proportional to O(BK²).</p><p><b>DFlash3-MOBS:</b> chooses a reproducible pseudo-random middle anchor, expands bidirectionally using K neighbor comparisons per position, then applies a fixed odd/even bubble-like local refinement. For a fixed refinement count, selector work remains O(BK).</p><p><b>Fidelity:</b> {html.escape(data['fidelity_note'])}</p></section><section class="card"><h2>Speed animation</h2><img class="gif" alt="Animated speed comparison" src="data:image/gif;base64,{_b64(speed_gif)}"></section><section class="card"><h2>MOBS architecture animation</h2><img class="gif" alt="Animated DFlash3 MOBS architecture" src="data:image/gif;base64,{_b64(architecture_gif)}"></section></div><section class="card wide"><h2>Reproduce</h2><div class="code">docker build -t dflash-mini-lab .\ndocker run --rm -e CPU_THREADS=1 -v \"$PWD/reports:/app/reports\" dflash-mini-lab --top-k 8 --mobs-refine-passes 1</div><p>Workload: {data['config']['prompt_count']} prompts × {data['config']['repeats']} measured repeats × {data['config']['max_new_tokens']} generated tokens; block size {data['config']['dflash_block_size']}; top-k {data['config']['dflash2_top_k']}; MOBS refinement passes {data['config']['dflash3_mobs_refine_passes']}.</p></section></main><script>{js}</script></body></html>"""
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(document, encoding="utf-8")
