from __future__ import annotations

from collections import defaultdict
import html
import json
from pathlib import Path
import statistics

import numpy as np

from .qwen_runtime import QwenDFlashRuntime


METHODS = ("normal_cached", "dflash_qwen", "dflash7_act")
DEFAULT_THRESHOLDS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)


def _read_prompts(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"{path} must contain a non-empty prompts list")
    return [str(prompt) for prompt in prompts]


def tune_v7_threshold(
    runtime: QwenDFlashRuntime,
    prompts: list[str],
    *,
    tokens: int = 8,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> tuple[float, list[dict]]:
    rows: list[dict] = []
    for threshold in thresholds:
        speeds: list[float] = []
        exact = True
        verify_counts: list[float] = []
        acceptances: list[float] = []
        for prompt in prompts:
            ids = runtime.encode(prompt)
            normal, _ = runtime.normal_cached_decode(ids, tokens)
            output, stats = runtime.dflash_decode(
                ids,
                tokens,
                method="dflash7_act",
                v7_margin_threshold=float(threshold),
            )
            exact = exact and bool(np.array_equal(normal, output))
            speeds.append(stats.tokens_per_second)
            verify_counts.append(stats.mean_verify_drafts)
            acceptances.append(stats.acceptance_rate)
        rows.append({
            "threshold": float(threshold),
            "median_tokens_per_second": float(statistics.median(speeds)),
            "mean_verify_drafts": float(statistics.mean(verify_counts)),
            "mean_acceptance_rate": float(statistics.mean(acceptances)),
            "all_exact": bool(exact),
        })
    valid = [row for row in rows if row["all_exact"]]
    if not valid:
        raise RuntimeError("no exact DFlash7 calibration setting")
    selected = max(valid, key=lambda row: row["median_tokens_per_second"])
    return float(selected["threshold"]), rows


def _summarize(records: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["method"]].append(record)
    normal_speed = statistics.median(row["tokens_per_second"] for row in grouped["normal_cached"])
    summary: dict[str, dict] = {}
    for method, rows in grouped.items():
        speed = statistics.median(row["tokens_per_second"] for row in rows)
        summary[method] = {
            "tokens_per_second_median": float(speed),
            "speedup_vs_normal": float(speed / normal_speed),
            "latency_ms_median": float(statistics.median(row["wall_seconds"] * 1000.0 for row in rows)),
            "prefill_ms_median": float(statistics.median(row["prefill_seconds"] * 1000.0 for row in rows)),
            "mean_target_forward_passes": float(statistics.mean(row["target_forward_passes"] for row in rows)),
            "mean_target_input_tokens": float(statistics.mean(row["target_input_tokens"] for row in rows)),
            "mean_tokens_per_target_pass": float(statistics.mean(row["tokens_per_target_pass"] for row in rows)),
            "mean_acceptance_rate": float(statistics.mean(row["acceptance_rate"] for row in rows)),
            "mean_verify_drafts": float(statistics.mean(row["mean_verify_drafts"] for row in rows)),
            "all_exact": bool(all(row["exact"] for row in rows)),
        }
    return summary


def _write_html(payload: dict, output_path: Path) -> None:
    rows = []
    labels = {
        "normal_cached": "Normal Qwen KV-cache",
        "dflash_qwen": "DFlash-Qwen hidden fusion",
        "dflash7_act": "DFlash7-ACT",
    }
    for method in METHODS:
        item = payload["summary"][method]
        rows.append(
            "<tr>"
            f"<td>{html.escape(labels[method])}</td>"
            f"<td>{item['tokens_per_second_median']:.3f}</td>"
            f"<td>{item['speedup_vs_normal']:.3f}×</td>"
            f"<td>{100.0 * item['mean_acceptance_rate']:.1f}%</td>"
            f"<td>{item['mean_target_forward_passes']:.2f}</td>"
            f"<td>{item['mean_target_input_tokens']:.2f}</td>"
            f"<td>{item['mean_verify_drafts']:.2f}</td>"
            f"<td>{'✓' if item['all_exact'] else '✗'}</td>"
            "</tr>"
        )
    calibration = "".join(
        "<tr>"
        f"<td>{row['threshold']:.2f}</td>"
        f"<td>{row['median_tokens_per_second']:.3f}</td>"
        f"<td>{100.0 * row['mean_acceptance_rate']:.1f}%</td>"
        f"<td>{row['mean_verify_drafts']:.2f}</td>"
        "</tr>"
        for row in payload["v7_calibration"]
    )
    output_path.write_text(
        f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Qwen3-0.6B DFlash7 benchmark</title>
<style>body{{font-family:system-ui;max-width:1100px;margin:40px auto;padding:0 20px}}table{{border-collapse:collapse;width:100%;margin:18px 0}}th,td{{padding:9px;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}code{{background:#f4f4f4;padding:2px 5px}}</style></head>
<body><h1>Qwen3-0.6B-Base — DFlash7-ACT real-model benchmark</h1>
<p>Frozen verifier: <code>{html.escape(payload['model']['id'])}</code>. Decode throughput excludes prefill. All speculative output is greedily verified by Qwen and compared token-for-token with normal cached decoding.</p>
<p><b>DFlash fidelity upgrades:</b> known verifier bonus-token anchor, selected multi-layer verifier hidden states, learned hidden fusion, bidirectional mask-slot decoder, target-head candidate rows, one-pass block verification, and rollback with Hugging Face DynamicCache.</p>
<table><thead><tr><th>Method</th><th>tok/s</th><th>speedup</th><th>draft acceptance</th><th>target calls</th><th>target input tokens</th><th>drafts / verify</th><th>exact</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>DFlash7 bounded calibration</h2><p>Selected margin threshold: <b>{payload['config']['v7_margin_threshold']:.2f}</b>. Threshold 0 is the fixed full-block DFlash behavior, so the search can fall back to the baseline.</p>
<table><thead><tr><th>margin threshold</th><th>tok/s</th><th>acceptance</th><th>drafts / verify</th></tr></thead><tbody>{calibration}</tbody></table>
<h2>Scope</h2><p>DFlash7-ACT is an experimental lab method. Its routing uses existing draft-logit margins to shorten uncertain suffixes; it adds no neural forward pass. The drafter is DFlash-inspired and much closer to the published mechanism than the earlier toy/LFM path, but it is not an official upstream checkpoint or training recipe.</p>
</body></html>""",
        encoding="utf-8",
    )


def run_benchmark(
    runtime: QwenDFlashRuntime,
    *,
    prompts_path: str | Path,
    calibration_prompts_path: str | Path,
    output_dir: str | Path,
    max_new_tokens: int = 12,
    repeats: int = 2,
    prompt_limit: int = 6,
    calibration_tokens: int = 8,
) -> dict:
    prompts = _read_prompts(prompts_path)[: max(1, int(prompt_limit))]
    calibration_prompts = _read_prompts(calibration_prompts_path)
    threshold, calibration = tune_v7_threshold(
        runtime,
        calibration_prompts,
        tokens=calibration_tokens,
    )
    records: list[dict] = []
    covered = 0
    cover_total = 0

    for repeat in range(max(1, int(repeats))):
        for prompt_index, prompt in enumerate(prompts):
            ids = runtime.encode(prompt)
            normal_output, normal_stats = runtime.normal_cached_decode(ids, max_new_tokens)
            for token in normal_output[1:].tolist():
                cover_total += 1
                covered += int(int(token) in runtime.candidate_set)
            normal_record = normal_stats.to_dict()
            normal_record.update(prompt=prompt, prompt_index=prompt_index, repeat=repeat, exact=True)
            records.append(normal_record)

            dflash_output, dflash_stats = runtime.dflash_decode(ids, max_new_tokens, method="dflash_qwen")
            dflash_record = dflash_stats.to_dict()
            dflash_record.update(
                prompt=prompt,
                prompt_index=prompt_index,
                repeat=repeat,
                exact=bool(np.array_equal(normal_output, dflash_output)),
            )
            records.append(dflash_record)

            v7_output, v7_stats = runtime.dflash_decode(
                ids,
                max_new_tokens,
                method="dflash7_act",
                v7_margin_threshold=threshold,
            )
            v7_record = v7_stats.to_dict()
            v7_record.update(
                prompt=prompt,
                prompt_index=prompt_index,
                repeat=repeat,
                exact=bool(np.array_equal(normal_output, v7_output)),
            )
            records.append(v7_record)

    summary = _summarize(records)
    payload = {
        "model": {
            "id": runtime.model_id,
            "target_parameter_count": int(runtime.target_parameter_count),
            "target_hidden_size": int(runtime.config.target_hidden_size),
            "target_vocab_size": int(runtime.config.target_vocab_size),
            "candidate_size": int(runtime.config.candidate_size),
            "aux_parameter_count": int(runtime.aux_parameter_count),
            "target_layer_ids": list(runtime.config.target_layer_ids),
        },
        "config": {
            "max_new_tokens": int(max_new_tokens),
            "repeats": int(repeats),
            "prompt_count": len(prompts),
            "block_size": int(runtime.config.block_size),
            "memory_tokens": int(runtime.config.memory_tokens),
            "v7_margin_threshold": float(threshold),
            "prefill_excluded_from_decode_timing": True,
            "cache": "Hugging Face DynamicCache with rollback crop after rejection",
            "anchor": "known greedy bonus token from previous verifier logits",
        },
        "mean_candidate_coverage": float(covered / max(cover_total, 1)),
        "v7_calibration": calibration,
        "summary": summary,
        "records": records,
        "training_metadata": runtime.metadata,
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_html(payload, out / "report.html")
    return payload
