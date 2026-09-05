from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import statistics
import time

import numpy as np

from .lfm_benchmark import LABELS, RealDecodeStats, _read_prompts, run_real_benchmark
from .lfm_runtime import LfmReferenceRuntime


SHOWCASE_METHODS = (
    "normal",
    "dflash",
    "dflash2",
    "dflash3_mobs",
    "dflash4_jump_mobs",
    "dflash5_fused_jump_mobs",
    "dflash6_boltzmann",
    "dflash6_bmobs",
    "dflash7_act",
)

SHOWCASE_LABELS = {
    **LABELS,
    "dflash7_act": "DFlash7-ACT",
}

ACT_THRESHOLDS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def _leading_confident_length(draft_logits: np.ndarray, threshold: float, limit: int) -> tuple[int, list[float]]:
    logits = np.asarray(draft_logits, dtype=np.float32)
    if logits.ndim != 2 or logits.shape[1] < 2:
        return max(1, min(int(limit), int(logits.shape[0]))), []
    top2 = np.partition(logits, kth=logits.shape[1] - 2, axis=1)[:, -2:]
    margins = np.max(top2, axis=1) - np.min(top2, axis=1)
    n = min(int(limit), int(margins.size))
    if threshold <= 0:
        return max(1, n), [float(x) for x in margins[:n]]
    keep = 0
    for margin in margins[:n]:
        if float(margin) < float(threshold):
            break
        keep += 1
    return max(1, keep), [float(x) for x in margins[:n]]


def dflash7_decode(
    runtime: LfmReferenceRuntime,
    input_ids: np.ndarray,
    max_new_tokens: int,
    *,
    margin_threshold: float,
):
    seq = np.asarray(input_ids, dtype=np.int64).copy()
    start_len = int(seq.size)
    target_calls = draft_calls = accepted_total = proposed_total = 0
    target_seconds = context_seconds = draft_seconds = selection_seconds = 0.0
    trimmed_total = 0
    raw_draft_total = 0
    all_margins: list[float] = []
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
        full_proposal = np.asarray(runtime.proposal_argmax(draft_logits), dtype=np.int64)
        raw_len = min(int(full_proposal.size), remaining)
        verify_len, margins = _leading_confident_length(draft_logits, float(margin_threshold), raw_len)
        verify_len = min(max(1, int(verify_len)), raw_len)
        proposal = full_proposal[:verify_len]
        raw_draft_total += raw_len
        trimmed_total += max(0, raw_len - verify_len)
        all_margins.extend(margins[:raw_len])
        selection_seconds += time.perf_counter() - t

        proposed_total += int(proposal.size)
        verify_input = np.concatenate([seq, proposal])
        t = time.perf_counter()
        logits = runtime.target_logits(verify_input)
        target_seconds += time.perf_counter() - t
        target_calls += 1

        p, k = int(seq.size), int(proposal.size)
        verifier = np.argmax(logits[p - 1 : p - 1 + k], axis=-1).astype(np.int64)
        mismatch = np.flatnonzero(proposal != verifier)
        accepted = k if mismatch.size == 0 else int(mismatch[0])
        accepted_total += accepted
        if accepted:
            seq = np.concatenate([seq, proposal[:accepted]])
        if accepted < k and int(seq.size) - start_len < int(max_new_tokens):
            seq = np.append(seq, verifier[accepted])

    seq = seq[: start_len + int(max_new_tokens)]
    wall = time.perf_counter() - wall0
    stats = RealDecodeStats(
        method="dflash7_act",
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
        "act_margin_threshold": float(margin_threshold),
        "act_raw_draft_tokens": int(raw_draft_total),
        "act_verified_draft_tokens": int(proposed_total),
        "act_trimmed_draft_tokens": int(trimmed_total),
        "act_mean_margin": float(statistics.fmean(all_margins)) if all_margins else 0.0,
        "act_mean_verified_drafts_per_block": float(proposed_total / max(draft_calls, 1)),
    }
    return seq, stats, meta


def _warm(runtime: LfmReferenceRuntime, prompt: str) -> None:
    ids = runtime.encode(prompt)
    _ = runtime.target_logits(ids)
    _ = runtime.draft_logits(runtime.context_features(ids))


def calibrate_act(
    runtime: LfmReferenceRuntime,
    prompts: list[str],
    *,
    tokens: int,
    thresholds: tuple[float, ...] = ACT_THRESHOLDS,
) -> tuple[float, list[dict]]:
    rows: list[dict] = []
    for threshold in thresholds:
        times: list[float] = []
        acceptances: list[float] = []
        verified: list[float] = []
        exact_flags: list[bool] = []
        for prompt in prompts:
            ids = runtime.encode(prompt)
            # Calibration reference is intentionally outside the benchmark prompts.
            ref = np.asarray(ids, dtype=np.int64).copy()
            for _ in range(int(tokens)):
                logits = runtime.target_logits(ref)
                ref = np.append(ref, int(np.argmax(logits[-1])))
            out, stats, meta = dflash7_decode(runtime, ids, int(tokens), margin_threshold=float(threshold))
            times.append(float(stats.wall_seconds))
            acceptances.append(float(stats.acceptance_rate))
            verified.append(float(meta["act_mean_verified_drafts_per_block"]))
            exact_flags.append(bool(np.array_equal(out, ref)))
        total_tokens = int(tokens) * len(prompts)
        wall = float(sum(times))
        rows.append(
            {
                "threshold": float(threshold),
                "tokens_per_second": float(total_tokens / max(wall, 1e-12)),
                "mean_acceptance_rate": float(statistics.fmean(acceptances)),
                "mean_verified_drafts_per_block": float(statistics.fmean(verified)),
                "all_exact": bool(all(exact_flags)),
            }
        )
    valid = [r for r in rows if r["all_exact"]]
    if not valid:
        raise RuntimeError("No exact DFlash7-ACT calibration setting")
    best = max(valid, key=lambda r: r["tokens_per_second"])
    return float(best["threshold"]), rows


def _benchmark_act(
    runtime: LfmReferenceRuntime,
    *,
    prompts: list[str],
    tokens: int,
    repeats: int,
    threshold: float,
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    output_rows: list[dict] = []
    for prompt_index, prompt in enumerate(prompts):
        ids = runtime.encode(prompt)
        reference = np.asarray(ids, dtype=np.int64).copy()
        for _ in range(int(tokens)):
            logits = runtime.target_logits(reference)
            reference = np.append(reference, int(np.argmax(logits[-1])))
        first_output = None
        for repeat in range(max(1, int(repeats))):
            output, stats, meta = dflash7_decode(runtime, ids, int(tokens), margin_threshold=float(threshold))
            exact = bool(np.array_equal(output, reference))
            row = stats.to_dict()
            row.update(meta)
            row.update(prompt=prompt, prompt_index=prompt_index, repeat=repeat, exact_match=exact)
            rows.append(row)
            if first_output is None:
                first_output = output
        output_rows.append(
            {
                "prompt": prompt,
                "normal": runtime.decode(reference),
                "dflash7_act": runtime.decode(first_output),
            }
        )
    return rows, output_rows


def _mean(rows: list[dict], key: str) -> float:
    return float(statistics.fmean(float(r[key]) for r in rows))


def _median(rows: list[dict], key: str) -> float:
    return float(statistics.median(float(r[key]) for r in rows))


def _augment_payload(base: dict, act_rows: list[dict], act_outputs: list[dict], threshold: float, calibration: list[dict]) -> dict:
    summary = base["summary"]
    act_summary = {
        "tokens_per_second_median": _median(act_rows, "tokens_per_second"),
        "latency_seconds_median": _median(act_rows, "wall_seconds"),
        "target_seconds_median": _median(act_rows, "target_seconds"),
        "draft_seconds_median": _median(act_rows, "draft_seconds"),
        "selection_seconds_median": _median(act_rows, "selection_seconds"),
        "mean_target_forward_passes": _mean(act_rows, "target_forward_passes"),
        "mean_draft_forward_passes": _mean(act_rows, "draft_forward_passes"),
        "mean_acceptance_rate": _mean(act_rows, "acceptance_rate"),
        "mean_tokens_per_target_pass": _mean(act_rows, "tokens_per_target_pass"),
        "mean_total_guidance_scores": 0.0,
        "mean_verified_drafts_per_block": _mean(act_rows, "act_mean_verified_drafts_per_block"),
        "mean_trimmed_draft_tokens": _mean(act_rows, "act_trimmed_draft_tokens"),
        "all_exact": all(bool(r["exact_match"]) for r in act_rows),
    }
    baseline = float(summary["normal"]["tokens_per_second_median"])
    act_summary["speedup_vs_normal"] = float(act_summary["tokens_per_second_median"] / max(baseline, 1e-12))
    summary["dflash7_act"] = act_summary
    base["runs"].extend(act_rows)
    base["config"]["dflash7_act_margin_threshold"] = float(threshold)
    base["config"]["benchmark_workload"] = "6 held-out prompts × 16 generated tokens × 2 repeats"
    base["dflash7_calibration"] = {
        "held_out_from_benchmark": True,
        "selected_threshold": float(threshold),
        "candidates": calibration,
    }
    base["method_scope"] = (
        "All LFM-compatible DFlash Mini Lab methods (Normal + DFlash1–7) use the same frozen "
        "LiquidAI/LFM2.5-350M-Base target. EAGLE-3 is not included in the same-target ranking because "
        "the available ready checkpoint is trained for Qwen3-1.7B, not LFM2.5."
    )
    base["showcase"] = {
        "version": 9,
        "live_demo_count": 2,
        "demo_note": "Browser demos animate measured benchmark behavior; they do not execute LFM inference client-side.",
        "act_outputs": act_outputs,
    }
    base["winner"] = max(SHOWCASE_METHODS, key=lambda m: summary[m]["tokens_per_second_median"])
    return base


def _seeded_demo_pattern(method: str, rate: float, n: int = 12) -> list[bool]:
    digest = hashlib.sha256(method.encode("utf-8")).digest()
    out: list[bool] = []
    threshold = int(max(0.0, min(1.0, rate)) * 255)
    for i in range(n):
        out.append(digest[i % len(digest)] <= threshold)
    return out


def build_showcase_report(data: dict, output_path: str | Path) -> None:
    summary = data["summary"]
    winner = data["winner"]
    methods = [m for m in SHOWCASE_METHODS if m in summary]
    rows = []
    for method in methods:
        s = summary[method]
        rows.append(
            f"<tr data-method='{method}'><th>{html.escape(SHOWCASE_LABELS[method])}</th>"
            f"<td>{s['tokens_per_second_median']:.3f}</td>"
            f"<td>{s['speedup_vs_normal']:.3f}×</td>"
            f"<td>{100*s['mean_acceptance_rate']:.1f}%</td>"
            f"<td>{s['mean_tokens_per_target_pass']:.2f}</td>"
            f"<td>{s['mean_target_forward_passes']:.2f}</td>"
            f"<td>{s['target_seconds_median']:.3f}s</td>"
            f"<td>{s['draft_seconds_median']:.3f}s</td>"
            f"<td>{s['selection_seconds_median']:.4f}s</td>"
            f"<td>{'✓' if s['all_exact'] else '✗'}</td></tr>"
        )

    cards = []
    for method in methods:
        s = summary[method]
        cards.append(
            f"<button class='method-card' data-pick='{method}'><span>{html.escape(SHOWCASE_LABELS[method])}</span>"
            f"<b>{s['speedup_vs_normal']:.3f}×</b><small>{s['tokens_per_second_median']:.3f} tok/s · {100*s['mean_acceptance_rate']:.1f}% accept</small></button>"
        )

    js_summary = {
        m: {
            "label": SHOWCASE_LABELS[m],
            "tps": float(summary[m]["tokens_per_second_median"]),
            "speedup": float(summary[m]["speedup_vs_normal"]),
            "acceptance": float(summary[m]["mean_acceptance_rate"]),
            "targetPasses": float(summary[m]["mean_target_forward_passes"]),
            "tokensPerPass": float(summary[m]["mean_tokens_per_target_pass"]),
            "targetSeconds": float(summary[m]["target_seconds_median"]),
            "draftSeconds": float(summary[m]["draft_seconds_median"]),
            "selectionSeconds": float(summary[m]["selection_seconds_median"]),
            "pattern": _seeded_demo_pattern(m, float(summary[m]["mean_acceptance_rate"])),
        }
        for m in methods
    }
    js_data = json.dumps(js_summary, separators=(",", ":"))
    model = data["model"]
    threshold = data["dflash7_calibration"]["selected_threshold"]
    calibration_rows = "".join(
        f"<tr><td>{r['threshold']:.2f}</td><td>{r['tokens_per_second']:.3f}</td><td>{100*r['mean_acceptance_rate']:.1f}%</td><td>{r['mean_verified_drafts_per_block']:.2f}</td><td>{'✓' if r['all_exact'] else '✗'}</td></tr>"
        for r in data["dflash7_calibration"]["candidates"]
    )

    css = r"""
    :root{--bg:#07111f;--panel:#0d1b2d;--panel2:#10233a;--text:#eef6ff;--muted:#91a5bd;--line:#203954;--cyan:#58d9ff;--lime:#7cf5b2;--amber:#ffc86b;--red:#ff7b8c;--violet:#b69cff;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:var(--text);background:var(--bg)}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:radial-gradient(circle at 15% -10%,#163452 0,transparent 34%),radial-gradient(circle at 100% 0,#1e254e 0,transparent 30%),var(--bg);line-height:1.55}.wrap{max-width:1240px;margin:auto;padding:28px 22px 80px}.hero{position:relative;overflow:hidden;border:1px solid #24415e;border-radius:26px;padding:34px;background:linear-gradient(145deg,#10283fdd,#0b1728e8);box-shadow:0 24px 70px #0008}.hero:after{content:"";position:absolute;inset:-100% 40% auto -20%;height:300px;background:linear-gradient(90deg,transparent,#58d9ff24,transparent);transform:rotate(18deg);animation:shine 7s linear infinite}@keyframes shine{to{transform:translateX(140%) rotate(18deg)}}h1{font-size:clamp(2.1rem,5vw,4.5rem);line-height:.95;margin:14px 0 18px;letter-spacing:-.05em}.kicker,.chip{display:inline-flex;align-items:center;gap:8px;border:1px solid #315473;border-radius:999px;padding:7px 11px;color:#b9eaff;background:#0e2235aa;font-size:.82rem;font-weight:750}.chips{display:flex;gap:8px;flex-wrap:wrap}.hero-grid{display:grid;grid-template-columns:1.45fr .85fr;gap:28px;align-items:end}.winner{border:1px solid #315472;border-radius:20px;padding:20px;background:#071626aa}.winner strong{font-size:2.2rem;color:var(--lime)}.metric-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:18px}.metric{border:1px solid var(--line);background:#0b192a;border-radius:17px;padding:16px}.metric b{display:block;font-size:1.6rem}.metric small,.muted{color:var(--muted)}section{margin-top:22px}.card{border:1px solid var(--line);border-radius:22px;background:linear-gradient(145deg,#0d1b2df5,#0a1727f5);padding:23px;box-shadow:0 16px 40px #0004}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:16px}table{width:100%;border-collapse:collapse;min-width:900px;font-size:.86rem}th,td{padding:11px 12px;text-align:right;border-bottom:1px solid #1d3249;white-space:nowrap}th:first-child,td:first-child{text-align:left}thead th{position:sticky;top:0;background:#11263d;color:#bcd3e8}tbody tr:hover{background:#12283f}.bars{display:grid;gap:10px}.bar-row{display:grid;grid-template-columns:165px 1fr 72px;align-items:center;gap:10px}.bar-track{height:15px;background:#081321;border-radius:999px;overflow:hidden;border:1px solid #203951}.bar-fill{height:100%;width:0;background:linear-gradient(90deg,var(--cyan),var(--lime));border-radius:999px;transition:width 1s cubic-bezier(.2,.8,.2,1)}.flow{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;align-items:center}.node{position:relative;min-height:92px;border:1px solid #294965;border-radius:16px;background:#0a1a2b;padding:14px;display:flex;align-items:center;justify-content:center;text-align:center;font-weight:750}.node.active{border-color:var(--cyan);box-shadow:0 0 0 1px #58d9ff66,0 0 28px #58d9ff28;transform:translateY(-2px)}.node:not(:last-child):after{content:"→";position:absolute;right:-16px;color:#5b7894;z-index:2}.demo{border:1px solid #294963;border-radius:20px;padding:20px;background:#081626}.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:12px 0}select,button{font:inherit}select,.btn{border:1px solid #345675;background:#10263b;color:var(--text);border-radius:12px;padding:9px 12px}.btn{cursor:pointer;font-weight:800}.btn.primary{background:linear-gradient(90deg,#1988b8,#2e9b70);border:0}.btn:hover{filter:brightness(1.12)}.race-lane{display:grid;grid-template-columns:150px 1fr 70px;gap:10px;align-items:center;margin:13px 0}.race-track{height:38px;border-radius:12px;background:#07111d;border:1px solid #1d354d;overflow:hidden;position:relative}.runner{height:100%;width:0;background:linear-gradient(90deg,#3a9bd1,#69e5b1);display:flex;align-items:center;justify-content:flex-end;padding-right:8px;font-weight:900;color:#05151e;transition:width .12s linear}.tokens{display:flex;gap:8px;flex-wrap:wrap;min-height:54px;margin-top:14px}.token{padding:9px 11px;border-radius:11px;border:1px solid #35546f;background:#10263a;opacity:0;transform:translateY(8px) scale(.96);transition:.25s}.token.show{opacity:1;transform:none}.token.accept{background:#123a2f;border-color:#36b77f;color:#bfffe0}.token.reject{background:#3b1c29;border-color:#d65572;color:#ffd0da}.token.correct{background:#3c3216;border-color:#d5a642;color:#ffe0a2}.method-picker{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.method-card{text-align:left;color:var(--text);border:1px solid #284761;border-radius:14px;background:#0a1a2a;padding:13px;cursor:pointer}.method-card:hover,.method-card.selected{border-color:var(--cyan);background:#102a40}.method-card span,.method-card b,.method-card small{display:block}.method-card b{font-size:1.45rem;color:var(--lime)}.method-card small{color:var(--muted)}.note{border-left:4px solid var(--amber);background:#2b2415;padding:13px 15px;border-radius:8px;color:#ffe8b8}.good{border-left-color:var(--lime);background:#102a22;color:#c7ffe1}code{background:#081522;border:1px solid #233b52;border-radius:6px;padding:2px 5px}.footer{color:#7e94ac;text-align:center;margin-top:34px;font-size:.85rem}@media(max-width:900px){.hero-grid,.grid2{grid-template-columns:1fr}.metric-grid{grid-template-columns:1fr 1fr}.flow{grid-template-columns:1fr 1fr}.node:after{display:none}.method-picker{grid-template-columns:1fr 1fr}}@media(max-width:580px){.metric-grid,.method-picker{grid-template-columns:1fr}.bar-row,.race-lane{grid-template-columns:100px 1fr 55px}.wrap{padding:14px 12px 50px}.hero,.card{padding:17px;border-radius:18px}}
    """

    doc = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><meta name='color-scheme' content='dark'><title>LFM2.5-350M · DFlash CPU Lab</title><style>{css}</style></head><body><main class='wrap'>
<section class='hero'><div class='chips'><span class='kicker'>LFM2.5-350M-Base</span><span class='chip'>CPU · float32 · 2 threads</span><span class='chip'>9 LFM-compatible modes</span><span class='chip'>exact greedy verification</span></div><div class='hero-grid'><div><h1>Speculation that can actually help a CPU.</h1><p class='muted'>A same-target, overhead-inclusive comparison of Normal decoding and DFlash1–7 on the frozen <code>{html.escape(model['id'])}</code> verifier. The report preserves wins and losses exactly as CI measures them.</p></div><div class='winner'><small>Fastest measured method</small><h2>{html.escape(SHOWCASE_LABELS[winner])}</h2><strong>{summary[winner]['speedup_vs_normal']:.3f}×</strong><div>{summary[winner]['tokens_per_second_median']:.3f} tok/s</div></div></div><div class='metric-grid'><div class='metric'><small>Target</small><b>{model['target_parameter_count']/1e6:.1f}M</b><span class='muted'>parameters</span></div><div class='metric'><small>Candidate coverage</small><b>{100*data['mean_candidate_coverage']:.1f}%</b><span class='muted'>held-out continuation</span></div><div class='metric'><small>ACT threshold</small><b>{threshold:g}</b><span class='muted'>held-out calibration</span></div><div class='metric'><small>Live demos</small><b>2</b><span class='muted'>data-driven animations</span></div></div></section>

<section class='card'><h2>Measured CPU ranking</h2><p class='muted'>The chart scales to the fastest method on this exact run. Hovering the table is useful for the overhead decomposition below.</p><div id='speedBars' class='bars'></div><div class='table-wrap' style='margin-top:18px'><table><thead><tr><th>Method</th><th>tok/s</th><th>vs Normal</th><th>Acceptance</th><th>Tokens/target pass</th><th>Target passes</th><th>Target time</th><th>Draft time</th><th>Select time</th><th>Exact</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>

<section class='card'><h2>Animated architecture</h2><p class='muted'>Choose a method below; the active stages change to show where its overhead comes from.</p><div class='method-picker'>{''.join(cards)}</div><div class='flow' style='margin-top:18px'><div class='node' data-stage='context'>Prompt + context</div><div class='node' data-stage='draft'>Parallel drafter</div><div class='node' data-stage='select'>Path selector / ACT</div><div class='node' data-stage='verify'>LFM target verify</div><div class='node' data-stage='accept'>Accept prefix</div><div class='node' data-stage='correct'>Correct + continue</div></div></section>

<section class='grid2'><div class='demo'><h2>Live demo 1 · decoding race</h2><p class='muted'>Pick any two measured methods and race 32 output tokens. Animation time is accelerated for the browser, but the ratio comes directly from measured tok/s.</p><div class='controls'><select id='raceA'></select><select id='raceB'></select><button class='btn primary' id='startRace'>Start race</button><button class='btn' id='resetRace'>Reset</button></div><div class='race-lane'><b id='raceALabel'></b><div class='race-track'><div class='runner' id='runnerA'></div></div><span id='raceAOut'>0/32</span></div><div class='race-lane'><b id='raceBLabel'></b><div class='race-track'><div class='runner' id='runnerB'></div></div><span id='raceBOut'>0/32</span></div><p class='note good' id='raceVerdict'>Select methods and start.</p></div>

<div class='demo'><h2>Live demo 2 · speculative block</h2><p class='muted'>Select a method and step through draft → verify → accept/correct. The accept/reject pattern is deterministic and parameterized by the method's measured acceptance rate.</p><div class='controls'><select id='blockMethod'></select><button class='btn primary' id='runBlock'>Run block</button><button class='btn' id='clearBlock'>Clear</button></div><div id='blockStage' class='muted'>Ready.</div><div id='tokenStrip' class='tokens'></div><p class='note' id='blockNote'>This is a mechanism animation using measured statistics, not model inference in your browser.</p></div></section>

<section class='grid2'><div class='card'><h2>Why CPU can win here</h2><p>LFM2.5-350M is small enough that a lightweight parallel drafter can be cheap, while verifying several positions in one target forward can still replace multiple sequential full-target passes. The useful quantity is not acceptance alone; it is <b>accepted work minus draft + selector + wider-verification overhead</b>.</p><div id='overheadBars' class='bars'></div></div><div class='card'><h2>DFlash7-ACT calibration</h2><p class='muted'>ACT is calibrated only on separate prompts. Threshold 0 means full-block DFlash behavior; larger thresholds trim uncertain suffixes earlier.</p><div class='table-wrap'><table style='min-width:560px'><thead><tr><th>Margin threshold</th><th>tok/s</th><th>Acceptance</th><th>Verified drafts/block</th><th>Exact</th></tr></thead><tbody>{calibration_rows}</tbody></table></div></div></section>

<section class='card'><h2>Compatibility boundary</h2><p><b>EAGLE-3 is intentionally not ranked here.</b> The ready EAGLE-3 checkpoint in Version 8 is trained for Qwen3-1.7B, not LFM2.5-350M. Running it against this target would not be a valid same-model benchmark. The Pages site links the separate EAGLE/Qwen report for cross-model context.</p><p class='note'>{html.escape(data['timing_note'])}</p></section>
<div class='footer'>Generated from GitHub Actions benchmark.json · Browser demos use measured metrics and never claim to run the model client-side.</div>
</main><script>const D={js_data};const methods=Object.keys(D);const $=id=>document.getElementById(id);const maxTps=Math.max(...methods.map(m=>D[m].tps));
function bars(){{const el=$('speedBars');el.innerHTML='';methods.forEach(m=>{{const r=document.createElement('div');r.className='bar-row';r.innerHTML=`<b>${{D[m].label}}</b><div class="bar-track"><div class="bar-fill" style="width:${{100*D[m].tps/maxTps}}%"></div></div><span>${{D[m].speedup.toFixed(3)}}×</span>`;el.appendChild(r)}})}}bars();
function opts(sel,selected){{sel.innerHTML=methods.map(m=>`<option value="${{m}}" ${{m===selected?'selected':''}}>${{D[m].label}}</option>`).join('')}}opts($('raceA'),'normal');opts($('raceB'),'{winner}');opts($('blockMethod'),'{winner}');
function selectMethod(m){{document.querySelectorAll('.method-card').forEach(x=>x.classList.toggle('selected',x.dataset.pick===m));const active=new Set(['context','verify','accept','correct']);if(m!=='normal')active.add('draft');if(!['normal','dflash'].includes(m))active.add('select');document.querySelectorAll('.node').forEach(n=>n.classList.remove('active'));let i=0;const nodes=[...document.querySelectorAll('.node')].filter(n=>active.has(n.dataset.stage));const timer=setInterval(()=>{{if(i)nodes[i-1].classList.remove('active');if(i>=nodes.length){{clearInterval(timer);return}}nodes[i++].classList.add('active')}},350);renderOverhead(m)}}document.querySelectorAll('.method-card').forEach(x=>x.onclick=()=>selectMethod(x.dataset.pick));selectMethod('{winner}');
function renderOverhead(m){{const x=D[m], parts=[['Target',x.targetSeconds],['Draft',x.draftSeconds],['Selection',x.selectionSeconds]],mx=Math.max(...parts.map(p=>p[1]),.000001);$('overheadBars').innerHTML=parts.map(p=>`<div class="bar-row"><b>${{p[0]}}</b><div class="bar-track"><div class="bar-fill" style="width:${{100*p[1]/mx}}%"></div></div><span>${{p[1].toFixed(3)}}s</span></div>`).join('')}}
let raceTimer=null;function resetRace(){{clearInterval(raceTimer);raceTimer=null;['A','B'].forEach(k=>{{$('runner'+k).style.width='0%';$('race'+k+'Out').textContent='0/32'}});$('raceVerdict').textContent='Select methods and start.'}}$('resetRace').onclick=resetRace;$('startRace').onclick=()=>{{resetRace();const a=$('raceA').value,b=$('raceB').value;$('raceALabel').textContent=D[a].label;$('raceBLabel').textContent=D[b].label;const scale=0.18/Math.max(D[a].tps,D[b].tps),start=performance.now(),total=32;raceTimer=setInterval(()=>{{const elapsed=(performance.now()-start)/1000,ta=Math.min(total,elapsed*D[a].tps/scale),tb=Math.min(total,elapsed*D[b].tps/scale);$('runnerA').style.width=(100*ta/total)+'%';$('runnerB').style.width=(100*tb/total)+'%';$('raceAOut').textContent=Math.floor(ta)+'/32';$('raceBOut').textContent=Math.floor(tb)+'/32';if(ta>=total||tb>=total){{clearInterval(raceTimer);const win=ta>=total?a:b;$('raceVerdict').textContent=`${{D[win].label}} reaches 32 tokens first. Measured relative speed: ${{D[win].speedup.toFixed(3)}}× Normal.`}}}},45)}};
function clearBlock(){{$('tokenStrip').innerHTML='';$('blockStage').textContent='Ready.'}}$('clearBlock').onclick=clearBlock;$('runBlock').onclick=()=>{{clearBlock();const m=$('blockMethod').value,x=D[m];selectMethod(m);if(m==='normal'){{$('blockStage').textContent='Normal decoding: one target pass produces one next token.';let i=0;const t=setInterval(()=>{{const e=document.createElement('span');e.className='token show accept';e.textContent='T'+(i+1);$('tokenStrip').appendChild(e);if(++i===4)clearInterval(t)}},420);return}}$('blockStage').textContent=`${{x.label}} drafts a block in parallel…`;const pattern=x.pattern.slice(0,4);const els=[];pattern.forEach((ok,i)=>{{const e=document.createElement('span');e.className='token';e.textContent='draft '+(i+1);$('tokenStrip').appendChild(e);els.push(e);setTimeout(()=>e.classList.add('show'),180*i)}});setTimeout(()=>{{$('blockStage').textContent='Target verifies the proposed block…';let rejected=false;els.forEach((e,i)=>{{const ok=!rejected&&pattern[i];if(ok)e.classList.add('accept');else{{rejected=true;e.classList.add('reject')}}}});setTimeout(()=>{{if(rejected){{const c=document.createElement('span');c.className='token show correct';c.textContent='target correction';$('tokenStrip').appendChild(c);$('blockStage').textContent='Accept the exact prefix, insert target correction, continue.'}}else $('blockStage').textContent='Whole proposed block accepted; continue from the verified suffix.';$('blockNote').textContent=`Measured ${{x.label}} acceptance: ${{(100*x.acceptance).toFixed(1)}}% · ${{x.tokensPerPass.toFixed(2)}} output tokens per target pass.`}},650)}},1100)}};
</script></body></html>"""
    Path(output_path).write_text(doc, encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    base = run_real_benchmark(
        aux_path=args.aux,
        prompts_path=args.prompts,
        output_dir=args.output_dir,
        model_id=args.model_id,
        max_new_tokens=args.tokens,
        repeats=args.repeats,
        prompt_limit=args.prompt_limit,
        top_k=args.top_k,
        jump_weight=args.jump_weight,
        fused_weight=args.fused_weight,
        fused_min_margin=args.fused_min_margin,
        boltzmann_temperature=args.boltzmann_temperature,
        bmobs_temperature=args.bmobs_temperature,
        cpu_threads=args.cpu_threads,
        dtype=args.dtype,
    )

    runtime = LfmReferenceRuntime(args.aux, model_id=args.model_id, cpu_threads=args.cpu_threads, dtype=args.dtype)
    benchmark_prompts = _read_prompts(args.prompts, args.prompt_limit)
    calibration_prompts = _read_prompts(args.calibration_prompts, args.calibration_prompt_limit)
    _warm(runtime, calibration_prompts[0])
    selected_threshold, calibration = calibrate_act(
        runtime,
        calibration_prompts,
        tokens=args.calibration_tokens,
    )
    _warm(runtime, benchmark_prompts[0])
    act_rows, act_outputs = _benchmark_act(
        runtime,
        prompts=benchmark_prompts,
        tokens=args.tokens,
        repeats=args.repeats,
        threshold=selected_threshold,
    )
    payload = _augment_payload(base, act_rows, act_outputs, selected_threshold, calibration)
    out = Path(args.output_dir)
    (out / "benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    build_showcase_report(payload, out / "report.html")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="LFM2.5-350M all-method CPU showcase")
    parser.add_argument("--aux", default="lfm-artifacts/lfm_aux.pt")
    parser.add_argument("--prompts", default="real_benchmarks/prompts.json")
    parser.add_argument("--calibration-prompts", default="real_benchmarks/calibration_prompts.json")
    parser.add_argument("--output-dir", default="lfm-reports")
    parser.add_argument("--model-id", default="LiquidAI/LFM2.5-350M-Base")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=2)
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
        "selected_act_threshold": payload["dflash7_calibration"]["selected_threshold"],
        "summary": payload["summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
