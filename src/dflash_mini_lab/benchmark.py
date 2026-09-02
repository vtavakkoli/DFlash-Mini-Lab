from __future__ import annotations

import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from .decoding import dflash2_decode, dflash3_mobs_decode, dflash_decode, normal_decode
from .runtime import CpuReferenceRuntime
from .tokenizer import WordTokenizer

METHODS = ("normal", "dflash", "dflash2", "dflash3_mobs")


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else 0.0


def _decode(method: str, runtime, ids, tokens, top_k, mobs_refine_passes):
    if method == "normal":
        return normal_decode(runtime, ids, tokens)
    if method == "dflash":
        return dflash_decode(runtime, ids, tokens)
    if method == "dflash2":
        return dflash2_decode(runtime, ids, tokens, top_k=top_k)
    return dflash3_mobs_decode(runtime, ids, tokens, top_k=top_k, refine_passes=mobs_refine_passes)


def run_benchmark(
    weights_path,
    tokenizer_path,
    prompts_path,
    out_json,
    max_new_tokens: int = 24,
    warmups: int = 1,
    repeats: int = 3,
    top_k: int = 4,
    mobs_refine_passes: int = 0,
) -> dict:
    runtime = CpuReferenceRuntime(weights_path)
    tok = WordTokenizer.load(tokenizer_path)
    prompts = json.loads(Path(prompts_path).read_text(encoding="utf-8"))["prompts"]
    warm_prompt = tok.encode(prompts[0])
    for _ in range(max(0, warmups)):
        for method in METHODS:
            _decode(method, runtime, warm_prompt, min(6, max_new_tokens), top_k, mobs_refine_passes)
    runs: list[dict] = []
    exactness: list[dict] = []
    for prompt_index, prompt in enumerate(prompts):
        ids = tok.encode(prompt)
        normal_reference, _ = normal_decode(runtime, ids, max_new_tokens)
        for repeat in range(repeats):
            for method in METHODS:
                output, stats = _decode(method, runtime, ids, max_new_tokens, top_k, mobs_refine_passes)
                row = stats.to_dict()
                row.update(prompt=prompt, prompt_index=prompt_index, repeat=repeat, exact_match=bool(np.array_equal(output, normal_reference)))
                runs.append(row)
        df_out, _ = dflash_decode(runtime, ids, max_new_tokens)
        d2_out, _ = dflash2_decode(runtime, ids, max_new_tokens, top_k=top_k)
        d3_out, _ = dflash3_mobs_decode(runtime, ids, max_new_tokens, top_k=top_k, refine_passes=mobs_refine_passes)
        exactness.append(
            {
                "prompt": prompt,
                "normal_equals_dflash": bool(np.array_equal(normal_reference, df_out)),
                "normal_equals_dflash2": bool(np.array_equal(normal_reference, d2_out)),
                "normal_equals_dflash3_mobs": bool(np.array_equal(normal_reference, d3_out)),
                "normal_text": tok.decode(normal_reference.tolist()),
            }
        )
    summary: dict[str, dict] = {}
    for method in METHODS:
        subset = [r for r in runs if r["method"] == method]
        tps = [float(r["tokens_per_second"]) for r in subset]
        lat = [float(r["latency_ms"]) for r in subset]
        summary[method] = {
            "tokens_per_second_mean": statistics.fmean(tps),
            "tokens_per_second_median": statistics.median(tps),
            "latency_ms_mean": statistics.fmean(lat),
            "latency_ms_median": statistics.median(lat),
            "latency_ms_p95": _percentile(lat, 95),
            "mean_target_forward_passes": statistics.fmean(float(r["target_forward_passes"]) for r in subset),
            "mean_draft_forward_passes": statistics.fmean(float(r["draft_forward_passes"]) for r in subset),
            "mean_acceptance_rate": statistics.fmean(float(r["acceptance_rate"]) for r in subset),
            "mean_tokens_per_target_pass": statistics.fmean(float(r["tokens_per_target_pass"]) for r in subset),
            "mean_selector_pair_scores": statistics.fmean(float(r["selector_pair_scores"]) for r in subset),
            "all_exact": all(bool(r["exact_match"]) for r in subset),
        }
    baseline = summary["normal"]["tokens_per_second_median"]
    for method in METHODS:
        summary[method]["speedup_vs_normal"] = summary[method]["tokens_per_second_median"] / max(baseline, 1e-12)
    d2_work = summary["dflash2"]["mean_selector_pair_scores"]
    d3_work = summary["dflash3_mobs"]["mean_selector_pair_scores"]
    summary["dflash3_mobs"]["selector_work_ratio_vs_dflash2"] = d3_work / max(d2_work, 1e-12)
    summary["dflash3_mobs"]["selector_work_reduction_vs_dflash2"] = 1.0 - d3_work / max(d2_work, 1e-12)
    summary["dflash3_mobs"]["throughput_ratio_vs_dflash2"] = summary["dflash3_mobs"]["tokens_per_second_median"] / max(summary["dflash2"]["tokens_per_second_median"], 1e-12)
    payload = {
        "schema_version": 2,
        "benchmark_name": "DFlash Mini Lab CPU reference benchmark",
        "generated_unix": int(time.time()),
        "backend": "NumPy float32 / BLAS CPU reference runtime",
        "fidelity_note": "Mechanism-level educational implementation. DFlash uses one-pass parallel block drafting plus lossless verifier correction. DFlash2 adds learned low-rank predecessor-conditioned top-k dynamic-programming path selection. DFlash3-MOBS uses a reproducible middle anchor and bidirectional O(B*K) neighbor selection; an optional fixed odd/even local refinement can be enabled for ablation. It does not claim binary equivalence with upstream GPU kernels/checkpoints.",
        "timing_note": "The tiny reference target recomputes the visible sequence on each target forward pass and does not implement a production KV cache. Use the results for reproducible relative algorithm study, not as production serving throughput.",
        "system": {"platform": platform.platform(), "python": platform.python_version(), "processor": platform.processor() or "unknown", "numpy": np.__version__, "cpu_threads_env": os.getenv("CPU_THREADS", "1")},
        "config": {
            "max_new_tokens": max_new_tokens,
            "warmups": warmups,
            "repeats": repeats,
            "prompt_count": len(prompts),
            "dflash_block_size": runtime.block_size,
            "dflash2_top_k": top_k,
            "dflash3_mobs_top_k": top_k,
            "dflash3_mobs_refine_passes": mobs_refine_passes,
        },
        "summary": summary,
        "exactness": exactness,
        "runs": runs,
    }
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
