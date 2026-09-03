from __future__ import annotations

import argparse
import json
import os

from .lfm_benchmark import run_real_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DFlash Mini Lab against a real frozen LFM2.5-350M-Base target")
    parser.add_argument("--aux", default="lfm-artifacts/lfm_aux.pt")
    parser.add_argument("--prompts", default="real_benchmarks/prompts.json")
    parser.add_argument("--output-dir", default="lfm-reports")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--prompt-limit", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--jump-weight", type=float, default=0.5)
    parser.add_argument("--fused-weight", type=float, default=1.0)
    parser.add_argument("--fused-min-margin", type=float, default=0.0)
    parser.add_argument("--boltzmann-temperature", type=float, default=0.15)
    parser.add_argument("--bmobs-temperature", type=float, default=0.35)
    parser.add_argument("--cpu-threads", type=int, default=int(os.getenv("CPU_THREADS", "2")))
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    args = parser.parse_args()
    payload = run_real_benchmark(
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
    print(json.dumps(payload["summary"], indent=2))
    print(f"Report: {args.output_dir}/report.html")


if __name__ == "__main__":
    main()
