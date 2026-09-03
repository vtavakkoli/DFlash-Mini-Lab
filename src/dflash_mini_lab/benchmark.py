from __future__ import annotations

import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from .decoding import (
    dflash2_decode,
    dflash3_mobs_decode,
    dflash4_jump_mobs_decode,
    dflash5_fused_jump_mobs_decode,
    dflash6_bmobs_decode,
    dflash6_boltzmann_decode,
    dflash_decode,
    normal_decode,
)
from .runtime import CpuReferenceRuntime
from .tokenizer import WordTokenizer

METHODS = (
    "normal",
    "dflash",
    "dflash2",
    "dflash3_mobs",
    "dflash4_jump_mobs",
    "dflash5_fused_jump_mobs",
    "dflash6_boltzmann",
    "dflash6_bmobs",
)


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else 0.0


def _decode(method, runtime, ids, tokens, top_k, mobs_refine_passes, jump_weight, fused_weight, fused_min_margin, boltzmann_temp, bmobs_temp):
    if method == "normal":
        return normal_decode(runtime, ids, tokens)
    if method == "dflash":
        return dflash_decode(runtime, ids, tokens)
    if method == "dflash2":
        return dflash2_decode(runtime, ids, tokens, top_k=top_k)
    if method == "dflash3_mobs":
        return dflash3_mobs_decode(runtime, ids, tokens, top_k=top_k, refine_passes=mobs_refine_passes)
    if method == "dflash4_jump_mobs":
        return dflash4_jump_mobs_decode(runtime, ids, tokens, top_k=top_k, jump_weight=jump_weight)
    if method == "dflash5_fused_jump_mobs":
        return dflash5_fused_jump_mobs_decode(runtime, ids, tokens, top_k=top_k, fused_weight=fused_weight, min_margin=fused_min_margin)
    if method == "dflash6_boltzmann":
        return dflash6_boltzmann_decode(runtime, ids, tokens, top_k=top_k, temperature=boltzmann_temp)
    return dflash6_bmobs_decode(runtime, ids, tokens, top_k=top_k, temperature=bmobs_temp)


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
    jump_weight: float = 0.5,
    fused_weight: float = 1.0,
    fused_min_margin: float = 0.0,
    boltzmann_temp: float = 0.15,
    bmobs_temp: float = 0.15,
) -> dict:
    runtime = CpuReferenceRuntime(weights_path)
    tok = WordTokenizer.load(tokenizer_path)
    prompts = json.loads(Path(prompts_path).read_text(encoding="utf-8"))["prompts"]
    warm_prompt = tok.encode(prompts[0])
    for _ in range(max(0, warmups)):
        for method in METHODS:
            _decode(method, runtime, warm_prompt, min(6, max_new_tokens), top_k, mobs_refine_passes, jump_weight, fused_weight, fused_min_margin, boltzmann_temp, bmobs_temp)

    runs: list[dict] = []
    exactness: list[dict] = []
    for prompt_index, prompt in enumerate(prompts):
        ids = tok.encode(prompt)
        normal_reference, _ = normal_decode(runtime, ids, max_new_tokens)
        for repeat in range(repeats):
            for method in METHODS:
                output, stats = _decode(method, runtime, ids, max_new_tokens, top_k, mobs_refine_passes, jump_weight, fused_weight, fused_min_margin, boltzmann_temp, bmobs_temp)
                row = stats.to_dict()
                row.update(prompt=prompt, prompt_index=prompt_index, repeat=repeat, exact_match=bool(np.array_equal(output, normal_reference)))
                runs.append(row)

        outputs = {
            "normal_equals_dflash": dflash_decode(runtime, ids, max_new_tokens)[0],
            "normal_equals_dflash2": dflash2_decode(runtime, ids, max_new_tokens, top_k=top_k)[0],
            "normal_equals_dflash3_mobs": dflash3_mobs_decode(runtime, ids, max_new_tokens, top_k=top_k, refine_passes=mobs_refine_passes)[0],
            "normal_equals_dflash4_jump_mobs": dflash4_jump_mobs_decode(runtime, ids, max_new_tokens, top_k=top_k, jump_weight=jump_weight)[0],
            "normal_equals_dflash5_fused_jump_mobs": dflash5_fused_jump_mobs_decode(runtime, ids, max_new_tokens, top_k=top_k, fused_weight=fused_weight, min_margin=fused_min_margin)[0],
            "normal_equals_dflash6_boltzmann": dflash6_boltzmann_decode(runtime, ids, max_new_tokens, top_k=top_k, temperature=boltzmann_temp)[0],
            "normal_equals_dflash6_bmobs": dflash6_bmobs_decode(runtime, ids, max_new_tokens, top_k=top_k, temperature=bmobs_temp)[0],
        }
        record = {"prompt": prompt, "normal_text": tok.decode(normal_reference.tolist())}
        record.update({key: bool(np.array_equal(normal_reference, output)) for key, output in outputs.items()})
        exactness.append(record)

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
            "mean_jump_forward_passes": statistics.fmean(float(r["jump_forward_passes"]) for r in subset),
            "mean_acceptance_rate": statistics.fmean(float(r["acceptance_rate"]) for r in subset),
            "mean_tokens_per_target_pass": statistics.fmean(float(r["tokens_per_target_pass"]) for r in subset),
            "mean_selector_pair_scores": statistics.fmean(float(r["selector_pair_scores"]) for r in subset),
            "mean_jump_candidate_scores": statistics.fmean(float(r["jump_candidate_scores"]) for r in subset),
            "mean_boltzmann_candidate_scores": statistics.fmean(float(r["boltzmann_candidate_scores"]) for r in subset),
            "mean_total_guidance_scores": statistics.fmean(float(r["total_guidance_scores"]) for r in subset),
            "mean_fused_anchor_uses": statistics.fmean(float(r["fused_anchor_uses"]) for r in subset),
            "all_exact": all(bool(r["exact_match"]) for r in subset),
        }

    baseline = summary["normal"]["tokens_per_second_median"]
    for method in METHODS:
        summary[method]["speedup_vs_normal"] = summary[method]["tokens_per_second_median"] / max(baseline, 1e-12)

    d2_work = summary["dflash2"]["mean_selector_pair_scores"]
    for method in ("dflash3_mobs", "dflash4_jump_mobs", "dflash5_fused_jump_mobs", "dflash6_bmobs"):
        work = summary[method]["mean_total_guidance_scores"]
        summary[method]["guidance_work_reduction_vs_dflash2"] = 1.0 - work / max(d2_work, 1e-12)

    for method in ("dflash6_boltzmann", "dflash6_bmobs"):
        summary[method]["throughput_ratio_vs_dflash"] = summary[method]["tokens_per_second_median"] / max(summary["dflash"]["tokens_per_second_median"], 1e-12)
        summary[method]["acceptance_delta_vs_dflash"] = summary[method]["mean_acceptance_rate"] - summary["dflash"]["mean_acceptance_rate"]

    payload = {
        "schema_version": 5,
        "benchmark_name": "DFlash Mini Lab CPU reference benchmark",
        "generated_unix": int(time.time()),
        "backend": "NumPy float32 / BLAS CPU reference runtime",
        "fidelity_note": "Mechanism-level educational implementation. DFlash6-Boltzmann and DFlash6-BMOBS are lab experiments, not upstream DFlash releases. They add deterministic context-seeded Gumbel/Boltzmann selection using existing draft logits only; BMOBS combines one Boltzmann middle anchor with O(B*K) gap filling. The target verifier still determines every emitted greedy token.",
        "timing_note": "The tiny reference target recomputes the visible sequence on each target forward pass and does not implement a production KV cache. Use the results for reproducible relative algorithm study, not production serving throughput.",
        "system": {
            "platform": platform.platform(), "python": platform.python_version(), "processor": platform.processor() or "unknown",
            "numpy": np.__version__, "cpu_threads_env": os.getenv("CPU_THREADS", "1"),
        },
        "config": {
            "max_new_tokens": max_new_tokens, "warmups": warmups, "repeats": repeats, "prompt_count": len(prompts),
            "dflash_block_size": runtime.block_size, "dflash2_top_k": top_k, "dflash3_mobs_top_k": top_k,
            "dflash3_mobs_refine_passes": mobs_refine_passes, "dflash4_jump_weight": jump_weight,
            "dflash5_fused_weight": fused_weight, "dflash5_fused_min_margin": fused_min_margin,
            "dflash6_boltzmann_top_k": top_k, "dflash6_boltzmann_temperature": boltzmann_temp,
            "dflash6_bmobs_top_k": top_k, "dflash6_bmobs_temperature": bmobs_temp,
        },
        "summary": summary,
        "exactness": exactness,
        "runs": runs,
    }
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
