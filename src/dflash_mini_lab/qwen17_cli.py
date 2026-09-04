from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .qwen17_benchmark import run_benchmark
from .qwen17_runtime_safe import Qwen17Runtime


def main() -> None:
    # Keep this real-model DFlash CLI in the Version 8 change set as an explicit
    # cross-regression gate: EAGLE-3 uses a separate dependency/runtime image,
    # while the earlier Qwen3-1.7B-Base DFlash study must remain independently
    # reproducible and exact after the new baseline is added.
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

    # qwen17_benchmark historically labeled the target as bf16 because the
    # distillation teacher is bf16. The measured verifier can intentionally use
    # float32 for strict token-for-token exactness, so persist the runtime dtype
    # supplied by the CLI rather than the training dtype.
    payload["model"]["dtype"] = args.dtype
    output_dir = Path(args.output_dir)
    (output_dir / "benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path = output_dir / "report.html"
    if report_path.exists():
        report = report_path.read_text(encoding="utf-8")
        report = report.replace(" · bf16 CPU · decode-only timing.", f" · {args.dtype} CPU · decode-only timing.")
        report_path.write_text(report, encoding="utf-8")

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
