# LFM2.5 All-14 GPU benchmark

The GPU path is an **additional local benchmark surface**. It does not replace the canonical GitHub Pages CPU evidence.

## What runs on the GPU

The CUDA runtime moves the two dominant inference components to the selected NVIDIA GPU:

- `LiquidAI/LFM2.5-350M-Base` target/verifier;
- the compact DFlash drafter.

The small routing/guidance components remain on CPU/NumPy so V1–V14 keep the same selection logic as the CPU study. Host/device transfer is included in measured wall time.

## Prerequisites

You need:

1. an NVIDIA GPU supported by your installed driver;
2. Docker with NVIDIA Container Toolkit configured;
3. Docker Compose v2 with GPU support.

A local CUDA toolkit is not required by the container. The image installs the PyTorch 2.10 CUDA 12.8 wheel and uses the host NVIDIA driver through the container runtime.

## One-command test

```bash
docker compose --profile gpu run --rm test-gpu
```

Old Compose syntax also works when available:

```bash
docker-compose --profile gpu run --rm test-gpu
```

The service first runs a real CUDA check:

- `torch.cuda.is_available()`;
- CUDA device count;
- device name, compute capability and memory;
- PyTorch/CUDA/cuDNN versions;
- a real FP16 CUDA matrix multiplication.

If CUDA is not visible, the command exits non-zero. It never silently falls back to CPU.

## First run

The first run prepares the same frozen LFM2.5 auxiliary artifacts used by the CPU benchmark if they do not already exist:

```text
gpu-artifacts/lfm_aux.pt
gpu-artifacts/lfm_dspark.pt
gpu-artifacts/v12_parareal.json
gpu-artifacts/v14_simple_parareal.json
```

These artifacts are persisted on the host and reused on later runs.

The preparation code remains CPU-compatible by design; the measured All-14 inference benchmark then moves the LFM target and DFlash drafter to CUDA.

## Outputs

```text
gpu-reports/index.html
gpu-reports/report.html
gpu-reports/benchmark.json
```

`benchmark.json` includes a `device` section with the detected GPU, compute capability, target dtype, PyTorch version and CUDA runtime version.

## Full benchmark defaults

The Compose service defaults to the same benchmark workload shape as the canonical study:

```text
6 held-out prompts
24 generated tokens
3 repeats
3 calibration prompts
8 calibration tokens
```

All 15 paths must exactly match the GPU normal-greedy reference.

## Fast smoke test

For a quick CUDA/integration test:

```bash
GPU_PROMPT_LIMIT=1 GPU_REPEATS=1 GPU_TOKENS=8 \
  docker compose --profile gpu run --rm test-gpu
```

The first invocation may still need to prepare artifacts. After artifacts exist, the smoke test is much faster.

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

Example:

```bash
LFM_GPU_DTYPE=float32 docker compose --profile gpu run --rm test-gpu
```

The target uses this dtype. The DFlash drafter remains float32.

## Select a GPU

```bash
CUDA_DEVICE=0 docker compose --profile gpu run --rm test-gpu
```

The container receives all GPUs through Compose; `CUDA_DEVICE` selects which device the runtime uses.

## TF32

For float32 CUDA runs, TF32 is enabled by default where supported:

```text
LFM_ALLOW_TF32=1
```

Disable it with:

```bash
LFM_ALLOW_TF32=0 docker compose --profile gpu run --rm test-gpu
```

## Interpreting GPU results

Do not compare the CPU and GPU speedup rankings as if they were the same experiment. The absolute target cost, PCIe transfer cost, kernel launch overhead and selector-to-target ratio all change on GPU.

The meaningful comparisons are:

- all methods inside the **same GPU run**;
- exactness against the **same GPU normal reference**;
- target calls, acceptance and tokens/target-call under the same device/dtype;
- CPU vs GPU absolute throughput only when the workload and artifacts are otherwise matched.

The current reference GPU runtime returns target logits to CPU because the existing exact verifier and calibration code consume NumPy arrays. That transfer cost is intentionally included. A future fully CUDA-native verifier could improve absolute GPU throughput further, but would be a different implementation condition.
