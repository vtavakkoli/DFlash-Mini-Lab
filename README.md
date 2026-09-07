# DFlash Mini Lab

A reproducible **LFM2.5 speculative-decoding research lab** centered on one target: `LiquidAI/LFM2.5-350M-Base`.

The canonical published study compares **14 speculative mechanisms** plus normal greedy decoding on CPU. A separate Docker Compose GPU path runs the same All-14 method logic with the LFM target and DFlash drafter on CUDA.

> [!IMPORTANT]
> DFlash3 through DFlash14 are experimental DFlash Mini Lab variants and are not upstream official DFlash releases.

## Canonical CPU result page

**https://vtavakkoli.github.io/DFlash-Mini-Lab/**

The GitHub Pages site is generated from the unified LFM2.5 CPU workflow and reports speedup, tokens/second, acceptance, target-forward count, tokens/target-call, selector/correction work, exactness, and an animated mechanism explorer. The machine-readable source of truth is `benchmark.json`.

## GPU test with Docker Compose

The repository now includes a CUDA-enabled local benchmark path.

### Prerequisites

- NVIDIA GPU + recent NVIDIA driver;
- Docker;
- NVIDIA Container Toolkit configured for Docker;
- Docker Compose v2 with GPU support.

### Run the full GPU benchmark

```bash
docker compose --profile gpu run --rm test-gpu
```

Legacy Compose syntax, when installed:

```bash
docker-compose --profile gpu run --rm test-gpu
```

The service does **not** silently fall back to CPU. It first checks:

- `torch.cuda.is_available()`;
- detected GPU(s), memory and compute capability;
- PyTorch/CUDA/cuDNN versions;
- a real FP16 matrix multiplication on CUDA.

If the check passes, the service prepares any missing LFM2.5 artifacts and runs Normal + all 14 speculative methods with exact greedy verification.

### Quick smoke test

```bash
GPU_PROMPT_LIMIT=1 GPU_REPEATS=1 GPU_TOKENS=8 \
  docker compose --profile gpu run --rm test-gpu
```

### GPU outputs

```text
gpu-reports/index.html
gpu-reports/report.html
gpu-reports/benchmark.json
```

Artifacts are persisted in:

```text
gpu-artifacts/
```

and reused on later runs.

Default target dtype is FP16:

```text
LFM_GPU_DTYPE=float16
```

Supported values are `float16`, `bfloat16`, and `float32`.

Example:

```bash
LFM_GPU_DTYPE=float32 CUDA_DEVICE=0 \
  docker compose --profile gpu run --rm test-gpu
```

The CUDA runtime moves the **LFM target/verifier and DFlash drafter** to GPU. Small MOBS/DSpark/Parareal routing remains CPU/NumPy so the selection logic stays aligned with the CPU implementation; host/device transfer cost is included in wall-clock measurements.

See [`docs/gpu.md`](docs/gpu.md) for the complete GPU protocol and interpretation rules.

## The 14 speculative methods

| # | Method | Core mechanism | Guidance cost |
|---:|---|---|---|
| 01 | DFlash | Parallel block argmax | parallel draft + verify |
| 02 | DFlash2 | Top-k predecessor-aware dynamic programming | O(BK²) |
| 03 | DFlash3-MOBS | Middle-out bidirectional selection | O(BK) |
| 04 | DFlash4-JUMP-MOBS | Sparse jump anchors + gap filling | O(BK + JK) + jump pass |
| 05 | DFlash5-FUSED-JUMP | Fused sparse residual anchors | O(BK + JKR) |
| 06 | DFlash6-Boltzmann | Deterministic confidence-adaptive exploration | O(BK) |
| 07 | DFlash6-BMOBS | Boltzmann middle anchor + MOBS | O(BK) |
| 08 | DFlash7-ACT | Adaptive verifier-suffix trimming | O(B) routing |
| 09 | V9 DSpark-Lite | Low-rank Markov correction + survival gate | O(BK × rank) |
| 10 | V10 Advanced Boltzmann | Sparse uncertainty routing | sparse O(BK) |
| 11 | V11 Boltzmann-Gated MOBS | Uncertainty-gated sparse MOBS | bounded O(BK) |
| 12 | V12 PARAREAL | Parallel affine fine-minus-coarse residual correction | O(RBKD) |
| 13 | **V13 MinOp** | **V10 policy with fused Torch top-2 + one-slot budget** | **O(B) routing after top-2** |
| 14 | **V14 Simple PARAREAL** | **Three-coefficient fine-gap estimator + one residual update** | **O(B) scalar correction** |

Normal autoregressive LFM2.5 is method `00` and defines the exact reference and `1.000×` speed baseline inside each matched run.

## V13 MinOp

V13 keeps the V10 decision policy while removing avoidable top-k plumbing:

```text
context
  │
  ▼
DFlash drafter
  │
  ├─ full logits stay in Torch
  └─ torch.topk(k=2)
       │
       ▼
  only B×2 IDs/scores
       │
       ▼
least-confident eligible slot only
       │
       ├─ V10 two-way decision
       └─ all other slots = top-1
       │
       ▼
LFM2.5 exact verify
```

V13 deliberately reuses the configuration selected for V10. This makes the V10→V13 comparison primarily an implementation/operation-cost experiment.

## V14 Simple PARAREAL

V14 compresses the Parareal state to the signed top1-top2 gap:

```text
G = draft_top1_logit - draft_top2_logit
F = target_logit(draft_top1) - target_logit(draft_top2)   # preparation only
```

It fits only three coefficients:

```text
F_hat = a + b·G + c·normalized_position
```

and performs one update:

```text
gap_1 = G + damping · (F_hat - G)
```

At inference, at most one position may switch from draft top-1 to draft top-2. The estimator is closed-form ridge regression; there is no neural correction model and no target-model call beyond the normal authoritative verifier.

See [`docs/version13-14.md`](docs/version13-14.md).

## Reproduce the canonical CPU study

Build the CPU image:

```bash
docker build -f Dockerfile.lfm -t dflash-lfm25 .
```

Prepare the common artifacts:

```bash
python -m dflash_mini_lab.lfm_prepare \
  --model-id LiquidAI/LFM2.5-350M-Base \
  --output-dir lfm-artifacts

python -m dflash_mini_lab.lfm_dspark \
  --aux lfm-artifacts/lfm_aux.pt \
  --output lfm-artifacts/lfm_dspark.pt \
  --seeds real_benchmarks/train_seeds.json

python -m dflash_mini_lab.v12_prepare \
  --aux lfm-artifacts/lfm_aux.pt \
  --output lfm-artifacts/v12_parareal.json \
  --seeds real_benchmarks/train_seeds.json

python -m dflash_mini_lab.v14_prepare \
  --aux lfm-artifacts/lfm_aux.pt \
  --output lfm-artifacts/v14_simple_parareal.json \
  --seeds real_benchmarks/train_seeds.json
```

Run Normal + all 14 speculative methods:

```bash
python -m dflash_mini_lab.lfm_all14_benchmark \
  --aux lfm-artifacts/lfm_aux.pt \
  --dspark lfm-artifacts/lfm_dspark.pt \
  --v12-model lfm-artifacts/v12_parareal.json \
  --v14-model lfm-artifacts/v14_simple_parareal.json \
  --prompts real_benchmarks/prompts.json \
  --calibration-prompts real_benchmarks/calibration_prompts.json \
  --output-dir lfm-reports \
  --tokens 24 --repeats 3 --prompt-limit 6 \
  --top-k 8 --cpu-threads 2
```

## Benchmark discipline

1. **One target model:** LFM2.5-350M-Base.
2. **Held-out benchmark prompts:** benchmark prompts are not used to fit V12 or V14.
3. **Separate calibration prompts:** ACT, V9, V10 and V11 calibrate outside the benchmark workload; V13 reuses V10.
4. **Rotated execution order:** method order changes across prompt/repeat combinations.
5. **Exactness gate:** all 15 paths must reproduce normal greedy output for the same backend/dtype.
6. **No result cherry-picking:** negative speedups are retained.
7. **Matched runtime:** comparisons are valid inside a matched CPU run or inside a matched GPU run. Do not mix the two rankings as if they were one experiment.

See [`docs/reproducibility.md`](docs/reproducibility.md).

## Repository layout

```text
Dockerfile.lfm                              canonical CPU image
Dockerfile.gpu                              CUDA 12.8 PyTorch image
docker-compose.yml                          benchmark + test-gpu services
scripts/test_gpu.sh                         one-command GPU check/prep/benchmark
src/dflash_mini_lab/lfm_runtime.py          CPU LFM2.5 reference runtime
src/dflash_mini_lab/lfm_gpu_runtime.py      CUDA target + DFlash drafter runtime
src/dflash_mini_lab/lfm_gpu_check.py        CUDA capability/matmul verification
src/dflash_mini_lab/lfm_gpu_benchmark.py    GPU All-14 benchmark entrypoint
src/dflash_mini_lab/lfm_all14_benchmark.py  canonical All-14 method/report logic
src/dflash_mini_lab/v13_minop.py            V13 MinOp
src/dflash_mini_lab/v14_simple_parareal.py  V14 scalar Parareal
docs/gpu.md                                 GPU protocol
.github/workflows/lfm-real-benchmark.yml     canonical CPU evidence workflow
```

## References

- Jian Chen, Yesheng Liang, Zhijian Liu. **DFlash: Block Diffusion for Flash Speculative Decoding.** arXiv:2602.06036, 2026.
- Official DFlash project: https://github.com/z-lab/dflash
- Vahid Tavakkoli et al. **Parareal Contribution to Speeding-Up the Solving of Nonlinear Ordinary Differential Equations on Parallel/Multi-Core Platforms for Sensing Systems.** The coarse/fine residual-correction principle motivates V12 and V14; these decoding variants are adaptations rather than claims of mathematical equivalence to the ODE solver.
- LiquidAI LFM2.5 model family.

## License

MIT.
