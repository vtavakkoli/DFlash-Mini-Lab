from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_RELEASE_TAG = "lfm25-all14-artifacts-v1"
DEFAULT_RELEASE_BASE = (
    "https://github.com/vtavakkoli/DFlash-Mini-Lab/releases/download/"
    + DEFAULT_RELEASE_TAG
)
ARTIFACT_NAMES = (
    "lfm_aux.pt",
    "lfm_dspark.pt",
    "v12_parareal.json",
    "v14_simple_parareal.json",
)
MANIFEST_NAME = "artifact-manifest.json"


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "DFlash-Mini-Lab/1.6.1"})
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(req, timeout=120) as response, tmp.open("wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            next_notice = 10
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total:
                    pct = int(done * 100 / total)
                    if pct >= next_notice:
                        print(f"  {target.name}: {pct}%")
                        next_notice += 10
        tmp.replace(target)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _release_base(args: argparse.Namespace) -> str:
    if args.artifact_base_url:
        return args.artifact_base_url.rstrip("/")
    tag = args.artifact_release_tag or DEFAULT_RELEASE_TAG
    return f"https://github.com/vtavakkoli/DFlash-Mini-Lab/releases/download/{tag}"


def ensure_artifacts(args: argparse.Namespace) -> dict:
    artifact_dir = Path(args.artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_dir / MANIFEST_NAME
    base = _release_base(args)

    if args.refresh_artifacts or not manifest_path.exists():
        print(f"Downloading artifact manifest from {base}")
        try:
            _download(f"{base}/{MANIFEST_NAME}", manifest_path)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise RuntimeError(
                    "Prepared GPU artifacts are not published at the configured GitHub Release yet. "
                    "Publish the repository artifact release first or pass --artifact-base-url to a mirror."
                ) from exc
            raise

    manifest = _load_json(manifest_path)
    files = manifest.get("files", {})
    if not isinstance(files, dict):
        raise RuntimeError("artifact manifest has no valid 'files' object")

    for name in ARTIFACT_NAMES:
        meta = files.get(name)
        if not isinstance(meta, dict) or not meta.get("sha256"):
            raise RuntimeError(f"artifact manifest is missing checksum metadata for {name}")
        target = artifact_dir / name
        expected_hash = str(meta["sha256"]).lower()
        expected_size = int(meta.get("size_bytes", 0) or 0)

        valid = target.exists()
        if valid and expected_size:
            valid = target.stat().st_size == expected_size
        if valid:
            print(f"Checking {name} ...", end=" ", flush=True)
            valid = _sha256(target) == expected_hash
            print("OK" if valid else "replace")

        if args.refresh_artifacts or not valid:
            print(f"Downloading {name} directly from GitHub Release")
            _download(f"{base}/{name}", target)
            actual_hash = _sha256(target)
            if actual_hash != expected_hash:
                target.unlink(missing_ok=True)
                raise RuntimeError(
                    f"SHA-256 mismatch for {name}: expected {expected_hash}, got {actual_hash}"
                )
            if expected_size and target.stat().st_size != expected_size:
                target.unlink(missing_ok=True)
                raise RuntimeError(f"size mismatch for {name}")
            print(f"  verified {name}: sha256={actual_hash[:16]}…")

    return manifest


def _torch_probe() -> dict | None:
    if importlib.util.find_spec("torch") is None:
        return None
    code = (
        "import json,torch; "
        "print(json.dumps({'version':torch.__version__,'cuda_build':torch.version.cuda," 
        "'cuda_available':bool(torch.cuda.is_available())}))"
    )
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
    need_torch = torch_info is None or not torch_info.get("cuda_build")

    if not missing and not need_torch:
        return

    details = []
    if missing:
        details.append("missing: " + ", ".join(missing))
    if need_torch:
        details.append("CUDA-enabled PyTorch 2.10 is missing")

    if args.no_bootstrap:
        raise RuntimeError(
            "; ".join(details)
            + ". Install the dependencies manually or rerun without --no-bootstrap."
        )

    print("Python dependencies need bootstrap (" + "; ".join(details) + ").")
    print("Installing only the required runtime dependencies; the repository itself is NOT installed.")

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
                "--upgrade",
                "--force-reinstall",
                "torch==2.10.0",
                "--index-url",
                "https://download.pytorch.org/whl/cu128",
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
        tune=args.tune,
        tuning_repeats=args.tuning_repeats,
        top_k=8,
        jump_weight=0.5,
        fused_weight=1.0,
        boltzmann_temperature=0.15,
        bmobs_temperature=0.35,
        cpu_threads=int(args.cpu_threads),
        dtype=args.dtype,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Zero-install DFlash Mini Lab GPU runner: download artifacts, verify CUDA, run All-14"
    )
    p.add_argument("--tune", action="store_true", help="Choose verification widths on separate calibration prompts")
    p.add_argument("--tuning-repeats", type=int, default=2)
    p.add_argument("--artifact-dir", default="gpu-artifacts")
    p.add_argument("--output-dir", default="gpu-reports")
    p.add_argument("--artifact-release-tag", default=DEFAULT_RELEASE_TAG)
    p.add_argument("--artifact-base-url", default=os.getenv("DFLASH_ARTIFACT_BASE_URL"))
    p.add_argument("--refresh-artifacts", action="store_true")
    p.add_argument("--download-only", action="store_true")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--smoke", action="store_true", help="1 prompt, 8 tokens, 1 repeat")
    p.add_argument("--no-bootstrap", action="store_true")
    p.add_argument("--tokens", type=int, default=24)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--prompt-limit", type=int, default=6)
    p.add_argument("--calibration-tokens", type=int, default=8)
    p.add_argument("--calibration-prompt-limit", type=int, default=3)
    p.add_argument("--cpu-threads", type=int, default=int(os.getenv("CPU_THREADS", "4")))
    p.add_argument(
        "--dtype",
        default=os.getenv("LFM_GPU_DTYPE", "float16"),
        choices=("float16", "bfloat16", "float32"),
    )
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

    from dflash_mini_lab import lfm_gpu_check, lfm_gpu_benchmark

    print("\n=== CUDA capability check ===")
    lfm_gpu_check.main()
    if args.check_only:
        return

    print("\n=== LFM2.5 All-14 GPU benchmark ===")
    data = lfm_gpu_benchmark.run(_benchmark_namespace(args))
    print(
        json.dumps(
            {
                "winner": data["winner"],
                "device": data.get("device", {}),
                "output_dir": str(Path(args.output_dir).resolve()),
                "summary": data["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
