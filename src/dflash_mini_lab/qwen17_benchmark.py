from __future__ import annotations

from collections import defaultdict
import html
import json
from pathlib import Path
import statistics

import numpy as np

from .qwen17_runtime import METHODS, Qwen17Runtime


METHOD_LABELS = {
    "normal_cached": "Normal Qwen KV-cache",
    "dflash": "DFlash hidden fusion",
    "dflash2": "DFlash2 DP selector",
    "dflash3_mobs": "DFlash3-MOBS",
    "dflash4_jump_mobs": "DFlash4-JUMP-MOBS",
    "dflash5_fused_jump_mobs": "DFlash5-FUSED-JUMP-MOBS",
    "dflash6_boltzmann": "DFlash6-Boltzmann",
    "dflash6_bmobs": "DFlash6-BMOBS",
    "dflash7_act": "DFlash7-ACT",
}

DEFAULT_SEARCH = {
    "jump_weight": (0.25, 0.5, 1.0),
    "fused_weight": (0.5, 1.0, 1.5),
    "boltzmann_temp": (0.05, 0.1, 0.2, 0.35),
    "bmobs_temp": (0.05, 0.1, 0.2, 0.35),
    "v7_margin_threshold": (0.0, 0.5, 1.0, 1.5, 2.0, 3.0),
}


def _read_prompts(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"{path} must contain a non-empty prompts list")
    return [str(x) for x in prompts]


def _run_method(runtime: Qwen17Runtime, ids: np.ndarray, tokens: int, method: str, params: dict):
    if method == "normal_cached":
        return runtime.normal_cached_decode(ids, tokens)
    return runtime.speculative_decode(ids, tokens, method=method, **params)


def tune_methods(runtime: Qwen17Runtime, prompts: list[str], *, tokens: int, top_k: int) -> tuple[dict, dict]:
    prompt_data = []
    for prompt in prompts:
        ids = runtime.encode(prompt)
        normal, _ = runtime.normal_cached_decode(ids, tokens)
        prompt_data.append((ids, normal))
    selected = {"top_k": int(top_k), "jump_weight": 0.5, "fused_weight": 1.0, "boltzmann_temp": 0.1, "bmobs_temp": 0.1, "v7_margin_threshold": 1.0}
    calibration = {}
    mapping = {
        "jump_weight": "dflash4_jump_mobs",
        "fused_weight": "dflash5_fused_jump_mobs",
        "boltzmann_temp": "dflash6_boltzmann",
        "bmobs_temp": "dflash6_bmobs",
        "v7_margin_threshold": "dflash7_act",
    }
    for param, method in mapping.items():
        rows = []
        for value in DEFAULT_SEARCH[param]:
            speeds=[]; accept=[]; target_tokens=[]; exact=True
            params={"top_k":int(top_k),param:float(value)}
            for ids,normal in prompt_data:
                output,stats=_run_method(runtime,ids,tokens,method,params)
                exact=exact and bool(np.array_equal(normal,output)); speeds.append(stats.tokens_per_second); accept.append(stats.acceptance_rate); target_tokens.append(stats.target_input_tokens)
            rows.append({"value":float(value),"median_tokens_per_second":float(statistics.median(speeds)),"mean_acceptance_rate":float(statistics.mean(accept)),"mean_target_input_tokens":float(statistics.mean(target_tokens)),"all_exact":bool(exact)})
        valid=[row for row in rows if row["all_exact"]]
        if not valid: raise RuntimeError(f"no exact calibration candidate for {method}")
        selected[param]=float(max(valid,key=lambda row:row["median_tokens_per_second"])["value"]); calibration[param]=rows
    return selected,calibration


def _summarize(records: list[dict]) -> dict:
    grouped=defaultdict(list)
    for record in records: grouped[record["method"]].append(record)
    normal_speed=statistics.median(row["tokens_per_second"] for row in grouped["normal_cached"]); summary={}
    for method in METHODS:
        rows=grouped[method]; speed=statistics.median(row["tokens_per_second"] for row in rows)
        summary[method]={
            "tokens_per_second_median":float(speed),"speedup_vs_normal":float(speed/normal_speed),
            "latency_ms_median":float(statistics.median(row["wall_seconds"]*1000 for row in rows)),
            "prefill_ms_median":float(statistics.median(row["prefill_seconds"]*1000 for row in rows)),
            "mean_target_forward_passes":float(statistics.mean(row["target_forward_passes"] for row in rows)),
            "mean_target_input_tokens":float(statistics.mean(row["target_input_tokens"] for row in rows)),
            "mean_tokens_per_target_pass":float(statistics.mean(row["tokens_per_target_pass"] for row in rows)),
            "mean_acceptance_rate":float(statistics.mean(row["acceptance_rate"] for row in rows)),
            "mean_verify_drafts":float(statistics.mean(row["mean_verify_drafts"] for row in rows)),
            "mean_selector_pair_scores":float(statistics.mean(row["selector_pair_scores"] for row in rows)),
            "mean_jump_candidate_scores":float(statistics.mean(row["jump_candidate_scores"] for row in rows)),
            "mean_fused_candidate_scores":float(statistics.mean(row["fused_candidate_scores"] for row in rows)),
            "mean_boltzmann_candidate_scores":float(statistics.mean(row["boltzmann_candidate_scores"] for row in rows)),
            "mean_total_guidance_scores":float(statistics.mean(row["total_guidance_scores"] for row in rows)),
            "all_exact":bool(all(row["exact"] for row in rows)),
        }
    return summary


def _write_html(payload: dict, output_path: Path) -> None:
    winner=payload["winner"]; rows=[]
    for method in METHODS:
        s=payload["summary"][method]; mark=" 🏆" if method==winner["method"] else ""
        rows.append("<tr>"+f"<td>{html.escape(METHOD_LABELS[method])}{mark}</td><td>{s['tokens_per_second_median']:.3f}</td><td>{s['speedup_vs_normal']:.3f}×</td><td>{100*s['mean_acceptance_rate']:.1f}%</td><td>{s['mean_target_forward_passes']:.2f}</td><td>{s['mean_target_input_tokens']:.2f}</td><td>{s['mean_tokens_per_target_pass']:.2f}</td><td>{s['mean_total_guidance_scores']:.0f}</td><td>{'✓' if s['all_exact'] else '✗'}</td></tr>")
    settings="".join(f"<li><code>{html.escape(str(k))}</code>: {v}</li>" for k,v in payload["selected_settings"].items()); train=payload["training_metadata"].get("training",{}); acc=train.get("drafter_position_accuracy",[]); accuracy_text=", ".join(f"{100*x:.1f}%" for x in acc) if acc else "n/a"
    output_path.write_text(f"""<!doctype html><html><head><meta charset='utf-8'><title>Qwen3-1.7B all DFlash methods</title><style>body{{font-family:system-ui,-apple-system,sans-serif;max-width:1250px;margin:36px auto;padding:0 20px;color:#171717}}h1,h2{{line-height:1.2}}.hero{{padding:18px 22px;background:#f5f6f8;border-radius:14px}}table{{border-collapse:collapse;width:100%;margin:20px 0;font-size:14px}}th,td{{padding:9px;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}code{{background:#f1f1f1;padding:2px 5px;border-radius:4px}}</style></head><body><h1>Qwen3-1.7B-Base — all DFlash lab methods</h1><div class='hero'><p><b>Winner:</b> {html.escape(METHOD_LABELS[winner['method']])} at <b>{winner['tokens_per_second']:.3f} tok/s</b> ({winner['speedup_vs_normal']:.3f}× normal cached Qwen).</p><p>Frozen target: <code>{html.escape(payload['model']['id'])}</code> · {payload['model']['target_parameter_count']:,} parameters · bf16 CPU · decode-only timing.</p><p>Training corpus: <b>{payload['training_metadata']['constructed_training_examples']:,} sliding blocks</b>, {payload['training_metadata']['training_scale_vs_qwen06']:.2f}× the earlier Qwen3-0.6B experiment. Candidate coverage: <b>{100*payload['mean_candidate_coverage']:.1f}%</b>.</p></div><table><thead><tr><th>Method</th><th>tok/s</th><th>vs normal</th><th>draft acceptance</th><th>target calls</th><th>target input tokens</th><th>tokens / target call</th><th>guidance scores</th><th>exact</th></tr></thead><tbody>{''.join(rows)}</tbody></table><h2>Selected calibration settings</h2><ul>{settings}</ul><h2>Drafter diagnostics</h2><p>Training position accuracies: {html.escape(accuracy_text)}. Benchmark prompts are held out from the distillation seed list.</p><h2>Method scope</h2><p>DFlash2 uses exact top-k dynamic programming over a learned context-gated low-rank predecessor scorer. MOBS uses the same scorer with middle-out/gap selection. DFlash4 adds a separate +2/+4 jump head. DFlash5 reuses drafter hidden states for fused sparse anchors. DFlash6-Boltzmann uses deterministic adaptive-temperature Gumbel selection; BMOBS samples one middle anchor then fills with MOBS. DFlash7-ACT shortens low-confidence suffixes before Qwen verification. Every proposal is greedily verified and required to reproduce normal cached Qwen token-for-token.</p><p><b>Fidelity note:</b> these are experimental mechanism-level adaptations in DFlash Mini Lab, not official upstream DFlash2–7 implementations or checkpoints.</p></body></html>""",encoding="utf-8")


def run_benchmark(runtime: Qwen17Runtime, *, prompts_path: str | Path, calibration_prompts_path: str | Path, output_dir: str | Path, max_new_tokens: int=16, repeats: int=1, prompt_limit: int=6, calibration_tokens: int=8, calibration_prompt_limit: int=2, top_k: int=8) -> dict:
    prompts=_read_prompts(prompts_path)[:max(1,int(prompt_limit))]; calibration_prompts=_read_prompts(calibration_prompts_path)[:max(1,int(calibration_prompt_limit))]
    selected,calibration=tune_methods(runtime,calibration_prompts,tokens=calibration_tokens,top_k=top_k)
    common={"top_k":int(top_k),"jump_weight":float(selected["jump_weight"]),"fused_weight":float(selected["fused_weight"]),"boltzmann_temp":float(selected["boltzmann_temp"]),"bmobs_temp":float(selected["bmobs_temp"]),"v7_margin_threshold":float(selected["v7_margin_threshold"])}
    records=[]; covered=cover_total=0; speculative=[m for m in METHODS if m!="normal_cached"]
    for repeat in range(max(1,int(repeats))):
        for prompt_index,prompt in enumerate(prompts):
            ids=runtime.encode(prompt); normal_output,normal_stats=runtime.normal_cached_decode(ids,max_new_tokens)
            for token in normal_output[1:].tolist(): cover_total+=1; covered+=int(int(token) in runtime.candidate_set)
            record=normal_stats.to_dict(); record.update(prompt=prompt,prompt_index=prompt_index,repeat=repeat,exact=True); records.append(record)
            shift=prompt_index%len(speculative); order=speculative[shift:]+speculative[:shift]
            for method in order:
                output,stats=runtime.speculative_decode(ids,max_new_tokens,method=method,**common); record=stats.to_dict(); record.update(prompt=prompt,prompt_index=prompt_index,repeat=repeat,exact=bool(np.array_equal(normal_output,output))); records.append(record)
    summary=_summarize(records); winner_method=max(METHODS,key=lambda m:summary[m]["tokens_per_second_median"]); winner={"method":winner_method,"tokens_per_second":summary[winner_method]["tokens_per_second_median"],"speedup_vs_normal":summary[winner_method]["speedup_vs_normal"]}
    payload={"schema_version":1,"model":{"id":runtime.model_id,"target_parameter_count":int(runtime.target_parameter_count),"target_hidden_size":int(runtime.config.target_hidden_size),"target_vocab_size":int(runtime.config.target_vocab_size),"candidate_size":int(runtime.config.candidate_size),"aux_parameter_count":int(runtime.aux_parameter_count),"target_layer_ids":list(runtime.config.target_layer_ids),"dtype":"bfloat16"},"config":{"max_new_tokens":int(max_new_tokens),"repeats":int(repeats),"prompt_count":len(prompts),"calibration_prompt_count":len(calibration_prompts),"calibration_tokens":int(calibration_tokens),"block_size":int(runtime.config.block_size),"memory_tokens":int(runtime.config.memory_tokens),"top_k":int(top_k),"prefill_excluded_from_decode_timing":True,"cache":"Hugging Face DynamicCache with rollback crop after rejection","anchor":"known greedy bonus token from previous verifier logits"},"selected_settings":selected,"calibration":calibration,"mean_candidate_coverage":float(covered/max(cover_total,1)),"summary":summary,"winner":winner,"records":records,"training_metadata":runtime.metadata}
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); (out/"benchmark.json").write_text(json.dumps(payload,indent=2),encoding="utf-8"); _write_html(payload,out/"report.html"); return payload
