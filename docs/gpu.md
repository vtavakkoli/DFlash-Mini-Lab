# LFM2.5 All-14 GPU benchmark

The GPU path is an **additional local benchmark surface**. It does not replace the canonical GitHub Pages CPU evidence.

## What runs on the GPU

The CUDA runtime moves the two dominant inference components to the selected NVIDIA GPU:

- `LiquidAI/LFM2.5-350M-Base` target/verifier;
- the compact DFlash drafter.

The small routing/guidance components remain on CPU/NumPy so V1–V14 keep the same selection logic as the CPU study. Host/device transfer is included in measured wall time.

## Preferred direct Python command

From the repository root:

```bash
python run_gpu.py
```

The repository package itself does not need to be installed. `run_gpu.py` inserts `src/` into `sys.path` before importing DFlash Mini Lab code.

If required runtime libraries are missing, the runner can install only the pinned runtime dependencies and restart itself. It never performs `pip install .` or `pip install -e .`.

Useful modes:

```bash
python run_gpu.py --smoke
python run_gpu.py --download-only
python run_gpu.py --check-only
```

`--smoke` uses one benchmark prompt, eight generated tokens and one repeat.

## Prepared artifacts are downloaded directly

The first run does not retrain the DFlash/DSpark/Parareal artifacts. Instead it downloads these individual files directly from the persistent GitHub Release `lfm25-all14-artifacts-v1`:

```text
lfm_aux.pt
lfm_dspark.pt
v12_parareal.json
v14_simple_parareal.json
artifact-manifest.json
```

The manifest records SHA-256 and byte size for every prepared file. `run_gpu.py` verifies both before loading the artifacts. Partial downloads use a `.part` file and are never accepted as valid artifacts.

Files are cached locally in:

```text
gpu-artifacts/
```

Force a new verified download with:

```bash
python run_gpu.py --refresh-artifacts
```

A custom artifact mirror can be selected with:

```bash
python run_gpu.py --artifact-base-url https://example.invalid/dflash-artifacts
```

or with the `DFLASH_ARTIFACT_BASE_URL` environment variable.

The target model weights are not redistributed by this release. Transformers still obtains `LiquidAI/LFM2.5-350M-Base` from its official Hugging Face source/cache.

## Direct-Python prerequisites

You need:

1. a supported NVIDIA GPU and driver;
2. Python 3.10+;
3. internet access on the first run for the release artifacts and, when not cached, the target model.

A local CUDA toolkit is not required by PyTorch wheels. If a CPU-only PyTorch installation is detected, the bootstrap path installs PyTorch 2.10.0 from the CUDA 12.8 wheel index. If a CUDA-enabled PyTorch build exists but the driver/GPU is not visible, the runner does not hide that error or silently fall back to CPU.

## Docker Compose

Docker remains available when you want a fully pinned container environment:

```bash
docker compose --profile gpu run --rm test-gpu
```

The Docker image uses the same `run_gpu.py` path. It installs PyTorch/Transformers dependencies in the image but does **not** install the DFlash Mini Lab repository as a package.

Docker prerequisites:

1. NVIDIA GPU + recent driver;
2. Docker;
3. NVIDIA Container Toolkit configured for Docker;
4. Docker Compose v2 with GPU support.

A local CUDA toolkit is not required by the container. The image installs the PyTorch 2.10 CUDA 12.8 wheel and uses the host NVIDIA driver through the container runtime.

The service runs a real CUDA check:

- `torch.cuda.is_available()`;
- CUDA device count;
- device name, compute capability and memory;
- PyTorch/CUDA/cuDNN versions;
- a real FP16 CUDA matrix multiplication.

If CUDA is not visible, the command exits non-zero. It never silently falls back to CPU.

## Outputs

```text
gpu-reports/index.html
gpu-reports/report.html
gpu-reports/benchmark.json
```

`benchmark.json` includes a `device` section with the detected GPU, compute capability, target dtype, PyTorch version and CUDA runtime version.

## Full benchmark defaults

The runner defaults to the same benchmark workload shape as the canonical study:

```text
6 held-out prompts
24 generated tokens
3 repeats
3 calibration prompts
8 calibration tokens
```

All 15 paths must exactly match the GPU normal-greedy reference.

## Dtype

Default:

```text
LFM_GPU_DTYPE=float16
```

Supported values:

```text
float16
bfloat16
float32
```

Examples:

```bash
python run_gpu.py --dtype float32
LFM_GPU_DTYPE=float32 docker compose --profile gpu run --rm test-gpu
```

The target uses this dtype. The DFlash drafter remains float32.

## Select a GPU

The CUDA runtime reads `CUDA_DEVICE`:

```bash
CUDA_DEVICE=0 python run_gpu.py
```

or:

```bash
CUDA_DEVICE=0 docker compose --profile gpu run --rm test-gpu
```

## TF32

For float32 CUDA runs, TF32 is enabled by default where supported:

```text
LFM_ALLOW_TF32=1
```

Disable it with:

```bash
LFM_ALLOW_TF32=0 python run_gpu.py --dtype float32
```

## Artifact publishing

`.github/workflows/publish-artifacts.yml` reproduces the frozen LFM2.5 auxiliary artifacts with the canonical CPU preparation environment, generates `artifact-manifest.json`, and publishes all five files as individual assets in the persistent `lfm25-all14-artifacts-v1` GitHub Release.

The release workflow is separate from the public CPU benchmark page so artifact distribution does not change the canonical CPU evidence protocol.

## Interpreting GPU results

Do not compare the CPU and GPU speedup rankings as if they were the same experiment. The absolute target cost, PCIe transfer cost, kernel launch overhead and selector-to-target ratio all change on GPU.

The meaningful comparisons are:

- all methods inside the **same GPU run**;
- exactness against the **same GPU normal reference**;
- target calls, acceptance and tokens/target-call under the same device/dtype;
- CPU vs GPU absolute throughput only when the workload and artifacts are otherwise matched.

The current reference GPU runtime returns target logits to CPU because the existing exact verifier and calibration code consume NumPy arrays. That transfer cost is intentionally included. A future fully CUDA-native verifier could improve absolute GPU throughput further, but would be a different implementation condition.
