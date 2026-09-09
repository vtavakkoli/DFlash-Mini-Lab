from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import lfm_all14_benchmark as all14
from .lfm_gpu_runtime import LfmGpuRuntime


def _gpu_report(data: dict) -> str:
    report = all14._report_html(data)
    info = data.get("device", {})
    report = report.replace("Canonical LFM2.5 CPU evidence", "LFM2.5 GPU evidence")
    report = report.replace("One target, one CPU protocol", "One target, one GPU protocol")
    old = f"{data['config']['cpu_threads']} CPU threads · {data['config']['dtype']}."
    gpu = (
        f"GPU: {info.get('device_name', 'CUDA')} · target {info.get('target_dtype', data['config']['dtype'])} · "
        f"CPU guidance threads {data['config']['cpu_threads']}."
    )
    report = report.replace(old, gpu)
    report = report.replace(
        "Speed claims apply only to the matched LFM2.5 benchmark shown here.",
        "GPU speed claims apply only to this matched CUDA run; the canonical CPU page remains a separate evidence surface.",
    )
    return report


def run(args: argparse.Namespace) -> dict:
    # all14.run owns the calibration, exactness and report data contract. Swap
    # only the runtime class so the GPU and CPU studies exercise identical
    # method logic.
    original_runtime = all14.LfmDSparkRuntime
    all14.LfmDSparkRuntime = LfmGpuRuntime
    try:
        data = all14.run(args)
    finally:
        all14.LfmDSparkRuntime = original_runtime

    info = dict(LfmGpuRuntime.last_device_info or {})
    data["study"] = "LFM2.5 All-14 GPU speculative decoding study"
    data["device"] = info
    data["config"]["backend"] = "cuda"
    data["config"]["gpu_device"] = info.get("device_name")
    data["config"]["gpu_target_dtype"] = info.get("target_dtype", args.dtype)
    data["config"]["gpu_target_and_drafter"] = True
    data["config"]["cpu_guidance"] = True

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = _gpu_report(data)
    (out / "report.html").write_text(report, encoding="utf-8")
    (out / "index.html").write_text(report, encoding="utf-8")
    return data


def main() -> None:
    p = argparse.ArgumentParser(description="Run Normal + 14 LFM2.5 methods with CUDA target/drafter")
    p.add_argument("--aux", default="lfm-artifacts/lfm_aux.pt")
    p.add_argument("--dspark", default="lfm-artifacts/lfm_dspark.pt")
    p.add_argument("--v12-model", default="lfm-artifacts/v12_parareal.json")
    p.add_argument("--v14-model", default="lfm-artifacts/v14_simple_parareal.json")
    p.add_argument("--prompts", default="real_benchmarks/prompts.json")
    p.add_argument("--calibration-prompts", default="real_benchmarks/calibration_prompts.json")
    p.add_argument("--output-dir", default="gpu-reports")
    p.add_argument("--tokens", type=int, default=int(os.getenv("GPU_TOKENS", "24")))
    p.add_argument("--repeats", type=int, default=int(os.getenv("GPU_REPEATS", "3")))
    p.add_argument("--prompt-limit", type=int, default=int(os.getenv("GPU_PROMPT_LIMIT", "6")))
    p.add_argument("--calibration-tokens", type=int, default=8)
    p.add_argument("--calibration-prompt-limit", type=int, default=3)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--jump-weight", type=float, default=0.5)
    p.add_argument("--fused-weight", type=float, default=1.0)
    p.add_argument("--boltzmann-temperature", type=float, default=0.15)
    p.add_argument("--bmobs-temperature", type=float, default=0.35)
    p.add_argument("--tune", action="store_true", help="Calibrate verification width per method on CUDA")
    p.add_argument("--tuning-repeats", type=int, default=2)
    p.add_argument("--cpu-threads", type=int, default=int(os.getenv("CPU_THREADS", "4")))
    p.add_argument(
        "--dtype",
        default=os.getenv("LFM_GPU_DTYPE", "float16"),
        choices=("float16", "bfloat16", "float32"),
    )
    args = p.parse_args()
    data = run(args)
    print(
        json.dumps(
            {
                "winner": data["winner"],
                "model": data["model"]["id"],
                "device": data.get("device", {}),
                "summary": data["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
