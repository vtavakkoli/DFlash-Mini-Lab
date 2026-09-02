from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", os.getenv("CPU_THREADS", "1"))
os.environ.setdefault("OMP_NUM_THREADS", os.getenv("CPU_THREADS", "1"))
os.environ.setdefault("MKL_NUM_THREADS", os.getenv("CPU_THREADS", "1"))

from .benchmark import run_benchmark
from .report import build_report
from .visuals import make_architecture_gif, make_speedup_gif


def main() -> None:
    p = argparse.ArgumentParser(description="CPU-only DFlash Mini Lab benchmark")
    p.add_argument("--weights", default="models/tiny_dflash_lab.npz")
    p.add_argument("--tokenizer", default="models/tokenizer.json")
    p.add_argument("--prompts", default="benchmarks/prompts.json")
    p.add_argument("--output-dir", default="reports")
    p.add_argument("--tokens", type=int, default=24)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--top-k", type=int, default=4)
    args = p.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    benchmark_json = out / "benchmark.json"
    speed_gif = out / "speedup.gif"
    architecture_gif = out / "architecture.gif"
    report_html = out / "report.html"
    payload = run_benchmark(args.weights, args.tokenizer, args.prompts, benchmark_json, args.tokens, args.warmups, args.repeats, args.top_k)
    make_speedup_gif(benchmark_json, speed_gif)
    make_architecture_gif(architecture_gif)
    build_report(benchmark_json, speed_gif, architecture_gif, report_html)
    print(json.dumps(payload["summary"], indent=2))
    print(f"Report: {report_html}")


if __name__ == "__main__":
    main()
