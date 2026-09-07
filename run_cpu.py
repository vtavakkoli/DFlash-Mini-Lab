from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Reuse the same release-asset downloader and checksum verification as GPU.
from run_gpu import DEFAULT_RELEASE_TAG, ensure_artifacts


def _torch_probe() -> dict | None:
    if importlib.util.find_spec("torch") is None:
        return None
    code = "import json,torch; print(json.dumps({'version':torch.__version__}))"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return None


def _restart_after_bootstrap() -> None:
    forwarded = [x for x in sys.argv[1:] if x != "--no-bootstrap"] + ["--no-bootstrap"]
    raise SystemExit(subprocess.call([sys.executable, str(Path(__file__).resolve()), *forwarded]))


def ensure_dependencies(args: argparse.Namespace) -> None:
    missing = [name for name in ("numpy", "transformers") if importlib.util.find_spec(name) is None]
    torch_info = _torch_probe()
    need_torch = torch_info is None

    if not missing and not need_torch:
        return

    details = []
    if missing:
        details.append("missing: " + ", ".join(missing))
    if need_torch:
        details.append("PyTorch 2.10 CPU runtime is missing")

    if args.no_bootstrap:
        raise RuntimeError(
            "; ".join(details)
            + ". Install the dependencies manually or rerun without --no-bootstrap."
        )

    print("Python dependencies need bootstrap (" + "; ".join(details) + ").")
    print("Installing runtime dependencies only; the repository itself is NOT installed.")

    if missing:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "numpy==2.3.5",
                "transformers==5.16.1",
            ]
        )

    if need_torch:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "torch==2.10.0",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ]
        )

    _restart_after_bootstrap()


def _benchmark_namespace(args: argparse.Namespace) -> argparse.Namespace:
    artifact_dir = Path(args.artifact_dir).resolve()
    tokens = 8 if args.smoke else args.tokens
    repeats = 1 if args.smoke else args.repeats
    prompt_limit = 1 if args.smoke else args.prompt_limit
    calibration_prompt_limit = 1 if args.smoke else args.calibration_prompt_limit
    return argparse.Namespace(
        aux=str(artifact_dir / "lfm_aux.pt"),
        dspark=str(artifact_dir / "lfm_dspark.pt"),
        v12_model=str(artifact_dir / "v12_parareal.json"),
        v14_model=str(artifact_dir / "v14_simple_parareal.json"),
        prompts=str(ROOT / "real_benchmarks" / "prompts.json"),
        calibration_prompts=str(ROOT / "real_benchmarks" / "calibration_prompts.json"),
        output_dir=str(Path(args.output_dir).resolve()),
        tokens=int(tokens),
        repeats=int(repeats),
        prompt_limit=int(prompt_limit),
        calibration_tokens=int(args.calibration_tokens),
        calibration_prompt_limit=int(calibration_prompt_limit),
        top_k=8,
        jump_weight=0.5,
        fused_weight=1.0,
        boltzmann_temperature=0.15,
        bmobs_temperature=0.35,
        cpu_threads=int(args.cpu_threads),
        dtype="float32",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Zero-install DFlash Mini Lab CPU runner: download artifacts and run canonical All-14"
    )
    p.add_argument("--artifact-dir", default="cpu-artifacts")
    p.add_argument("--output-dir", default="cpu-reports")
    p.add_argument("--artifact-release-tag", default=DEFAULT_RELEASE_TAG)
    p.add_argument("--artifact-base-url", default=os.getenv("DFLASH_ARTIFACT_BASE_URL"))
    p.add_argument("--refresh-artifacts", action="store_true")
    p.add_argument("--download-only", action="store_true")
    p.add_argument("--smoke", action="store_true", help="1 prompt, 8 tokens, 1 repeat")
    p.add_argument("--no-bootstrap", action="store_true")
    p.add_argument("--tokens", type=int, default=24)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--prompt-limit", type=int, default=6)
    p.add_argument("--calibration-tokens", type=int, default=8)
    p.add_argument("--calibration-prompt-limit", type=int, default=3)
    p.add_argument("--cpu-threads", type=int, default=int(os.getenv("CPU_THREADS", "2")))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    manifest = ensure_artifacts(args)
    print(
        "Artifacts ready:",
        manifest.get("model_id", "LiquidAI/LFM2.5-350M-Base"),
        "source",
        manifest.get("source_sha", "unknown"),
    )
    if args.download_only:
        return

    ensure_dependencies(args)

    from dflash_mini_lab import lfm_all14_benchmark

    print("\n=== LFM2.5 All-14 CPU benchmark ===")
    data = lfm_all14_benchmark.run(_benchmark_namespace(args))
    print(
        json.dumps(
            {
                "winner": data["winner"],
                "model": data["model"]["id"],
                "backend": "cpu",
                "output_dir": str(Path(args.output_dir).resolve()),
                "summary": data["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
