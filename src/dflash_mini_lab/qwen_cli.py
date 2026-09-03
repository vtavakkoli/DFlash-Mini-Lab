from __future__ import annotations

import argparse
import json
import os

from .qwen_benchmark import run_benchmark
from .qwen_runtime import QwenDFlashRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DFlash and DFlash7-ACT on Qwen3-0.6B-Base")
    parser.add_argument("--aux", default="qwen-artifacts/qwen_dflash.pt")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--prompts", default="qwen_benchmarks/prompts.json")
    parser.add_argument("--calibration-prompts", default="qwen_benchmarks/calibration_prompts.json")
    parser.add_argument("--output-dir", default="qwen-reports")
    parser.add_argument("--tokens", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--prompt-limit", type=int, default=6)
    parser.add_argument("--calibration-tokens", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=int(os.getenv("CPU_THREADS", "2")))
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    args = parser.parse_args()

    runtime = QwenDFlashRuntime(
        args.aux,
        model_id=args.model_id,
        cpu_threads=args.cpu_threads,
        dtype=args.dtype,
    )
    payload = run_benchmark(
        runtime,
        prompts_path=args.prompts,
        calibration_prompts_path=args.calibration_prompts,
        output_dir=args.output_dir,
        max_new_tokens=args.tokens,
        repeats=args.repeats,
        prompt_limit=args.prompt_limit,
        calibration_tokens=args.calibration_tokens,
    )
    print(json.dumps({
        "model": payload["model"],
        "config": payload["config"],
        "mean_candidate_coverage": payload["mean_candidate_coverage"],
        "summary": payload["summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
