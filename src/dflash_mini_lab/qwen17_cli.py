from __future__ import annotations

import argparse
import json
import os

from .qwen17_benchmark import run_benchmark
from .qwen17_runtime_safe import Qwen17Runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark all DFlash Mini Lab methods on Qwen3-1.7B-Base")
    parser.add_argument("--aux", default="qwen17-artifacts/qwen17_all_methods.pt")
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--prompts", default="qwen_benchmarks/prompts.json")
    parser.add_argument("--calibration-prompts", default="qwen_benchmarks/calibration_prompts.json")
    parser.add_argument("--output-dir", default="qwen17-reports")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--prompt-limit", type=int, default=6)
    parser.add_argument("--calibration-tokens", type=int, default=8)
    parser.add_argument("--calibration-prompt-limit", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=int(os.getenv("CPU_THREADS", "2")))
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    args = parser.parse_args()
    runtime = Qwen17Runtime(args.aux, model_id=args.model_id, cpu_threads=args.cpu_threads, dtype=args.dtype)
    payload = run_benchmark(
        runtime,
        prompts_path=args.prompts,
        calibration_prompts_path=args.calibration_prompts,
        output_dir=args.output_dir,
        max_new_tokens=args.tokens,
        repeats=args.repeats,
        prompt_limit=args.prompt_limit,
        calibration_tokens=args.calibration_tokens,
        calibration_prompt_limit=args.calibration_prompt_limit,
        top_k=args.top_k,
    )
    print(json.dumps({
        "model": payload["model"],
        "training_scale": payload["training_metadata"].get("training_scale_vs_qwen06"),
        "candidate_coverage": payload["mean_candidate_coverage"],
        "selected_settings": payload["selected_settings"],
        "winner": payload["winner"],
        "summary": payload["summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
