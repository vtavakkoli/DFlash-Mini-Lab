"""Interleave legacy, optimized and calibrated decoding on one loaded model.

First run run_cpu.py --tune (or run_gpu.py --tune) to create the settings file.
No training or configuration selection is performed on the measured prompts.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import transformers

from dflash_mini_lab.lfm_all14_benchmark import METHODS, _run_method, _greedy_reference
from dflash_mini_lab.lfm_all12_benchmark import _v12_config
from dflash_mini_lab.lfm_benchmark import _read_prompts
from dflash_mini_lab.lfm_dspark import LfmDSparkRuntime
from dflash_mini_lab.lfm_v10 import V10Config
from dflash_mini_lab.v11_boltzmann_mobs import V11Config
from dflash_mini_lab.v12_parareal import load_linear_model
from dflash_mini_lab.v14_simple_parareal import load_estimator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", required=True, help="benchmark.json from a --tune run on this backend")
    parser.add_argument("--artifact-dir", default="cpu-artifacts")
    parser.add_argument("--output", default="cpu-reports/optimization-comparison.json")
    parser.add_argument("--prompts", default=str(ROOT / "real_benchmarks/prompts.json"))
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prompt-limit", type=int, default=6)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    if min(args.tokens, args.repeats, args.prompt_limit, args.cpu_threads) < 1:
        parser.error("tokens, repeats, prompt limit and CPU threads must be positive")
    settings = json.loads(Path(args.settings).read_text())
    selected = settings["selected_settings"]
    if not selected.get("verify_widths"):
        parser.error("settings must come from a --tune run")
    if (settings["config"].get("backend") == "cuda") != args.gpu:
        parser.error("settings and comparison must use the same CPU/CUDA backend")
    if settings["config"]["cpu_threads"] != args.cpu_threads:
        parser.error("use the same CPU thread count as the settings run")
    cls = LfmDSparkRuntime
    if args.gpu:
        from dflash_mini_lab.lfm_gpu_runtime import LfmGpuRuntime
        cls = LfmGpuRuntime
    artifact = Path(args.artifact_dir)
    runtime = cls(artifact / "lfm_aux.pt", artifact / "lfm_dspark.pt",
                  cpu_threads=args.cpu_threads, dtype=settings["config"]["dtype"])
    if runtime.model_id != settings["model"]["id"]:
        parser.error("artifact target does not match the settings file")
    if args.gpu and runtime.gpu_dtype_name != settings["config"]["dtype"]:
        parser.error("GPU dtype does not match the settings file")
    v12_model = load_linear_model(artifact / "v12_parareal.json")
    options = dict(
        top_k=settings["config"]["top_k"],
        jump_weight=settings["config"].get("jump_weight", 0.5),
        fused_weight=settings["config"].get("fused_weight", 1.0),
        boltzmann_temperature=settings["config"].get("boltzmann_temperature", 0.15),
        bmobs_temperature=settings["config"].get("bmobs_temperature", 0.35),
        act_threshold=selected["act_margin_threshold"],
        dspark_floor=selected["dspark_survival_floor"],
        v10_config=V10Config(**selected["v10"]), v11_config=V11Config(**selected["v11"]),
        v12_model=v12_model, v12_config=_v12_config(v12_model, settings["config"]["top_k"]),
        v14_estimator=load_estimator(artifact / "v14_simple_parareal.json"),
    )
    modes = ("legacy", "optimized", "tuned")
    def configure(mode):
        runtime.optimize_target_logits = mode != "legacy"
        runtime.use_bonus_token = mode != "legacy"
        runtime.verify_widths = selected["verify_widths"] if mode == "tuned" else {}
    prompts = _read_prompts(args.prompts, args.prompt_limit)
    if not prompts:
        parser.error("prompts must be non-empty")
    ids = runtime.encode(prompts[0])
    for mode in modes:
        configure(mode)
        for method in METHODS:
            _run_method(method, runtime, ids, min(args.tokens, runtime.block_size + 1), **options)

    rows = []
    for pi, prompt in enumerate(prompts):
        ids = runtime.encode(prompt)
        reference = _greedy_reference(runtime, ids, args.tokens)
        for repeat in range(args.repeats):
            shift = (pi * args.repeats + repeat) % len(METHODS)
            for mi, method in enumerate(METHODS[shift:] + METHODS[:shift]):
                offset = (pi * args.repeats + repeat + mi) % len(modes)
                for mode in modes[offset:] + modes[:offset]:
                    configure(mode)
                    output, stats, metadata = _run_method(method, runtime, ids, args.tokens, **options)
                    row = stats.to_dict()
                    row.update(metadata)
                    row.update(mode=mode, prompt_index=pi, repeat=repeat,
                               exact_match=bool(np.array_equal(output, reference)))
                    rows.append(row)
        print(f"Finished prompt {pi + 1}/{len(prompts)}", flush=True)
    summary = {}
    for method in METHODS:
        per_mode = {}
        legacy = {(r["prompt_index"], r["repeat"]): r["wall_seconds"]
                  for r in rows if r["method"] == method and r["mode"] == "legacy"}
        for mode in modes:
            samples = [r for r in rows if r["method"] == method and r["mode"] == mode]
            per_mode[mode] = {
                "tokens_per_second": args.tokens * len(samples) / sum(r["wall_seconds"] for r in samples),
                "paired_speedup_median": statistics.median(
                    legacy[(r["prompt_index"], r["repeat"])] / r["wall_seconds"] for r in samples),
                "mean_target_calls": statistics.fmean(r["target_forward_passes"] for r in samples),
                "all_exact": all(r["exact_match"] for r in samples),
            }
        summary[method] = per_mode
    data = {
        "description": "Interleaved legacy replay, suffix projection plus bonus tokens, and calibrated verification widths",
        "legacy_replay": "Full target logits, no bonus token, full draft width; same frozen selector configurations in all modes",
        "model": runtime.model_id,
        "environment": {"platform": platform.platform(), "torch": torch.__version__,
                        "transformers": transformers.__version__, "device": getattr(runtime, "device_info", {"backend": "cpu"})},
        "config": vars(args), "source_settings": settings["selected_settings"],
        "prompts": prompts, "summary": summary, "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not all(row["exact_match"] for row in rows):
        raise SystemExit("Exactness failed; see diagnostic JSON")


if __name__ == "__main__":
    main()
