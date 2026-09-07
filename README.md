# DFlash Mini Lab

A reproducible **LFM2.5 speculative-decoding research lab** centered on one target: `LiquidAI/LFM2.5-350M-Base`.

The canonical published study compares **14 speculative mechanisms** plus normal greedy decoding on CPU. Separate Docker Compose CPU and GPU paths run the same All-14 method logic with exact greedy verification.

> [!IMPORTANT]
> DFlash3 through DFlash14 are experimental DFlash Mini Lab variants and are not upstream official DFlash releases.

## Canonical CPU result page

**https://vtavakkoli.github.io/DFlash-Mini-Lab/**

The GitHub Pages site is generated from the unified LFM2.5 CPU workflow and reports speedup, tokens/second, acceptance, target-forward count, tokens/target-call, selector/correction work, exactness, and an animated mechanism explorer. The machine-readable source of truth is `benchmark.json`.

## Zero-install local runners

A fresh clone does **not** need `pip install .` or `pip install -e .`.

CPU:

```bash
python run_cpu.py
```

GPU:

```bash
python run_gpu.py
```

The runners add `src/` to `sys.path` themselves, download prepared artifacts from the persistent GitHub Release `lfm25-all14-artifacts-v1`, and verify every artifact by SHA-256 and byte size before use.

Downloaded artifacts:

```text
lfm_aux.pt
lfm_dspark.pt
v12_parareal.json
v14_simple_parareal.json
artifact-manifest.json
```

The target model weights are not redistributed; `LiquidAI/LFM2.5-350M-Base` is still loaded from Hugging Face.

Useful commands:

```bash
python run_cpu.py --smoke
python run_cpu.py --download-only
python run_gpu.py --smoke
python run_gpu.py --download-only
python run_gpu.py --check-only
python run_gpu.py --refresh-artifacts
```

If required runtime dependencies are missing, the direct runners can install only those dependencies and restart themselves. The repository package itself is never installed.

## CPU test with Docker Compose

Run the full canonical CPU benchmark with:

```bash
docker compose --profile cpu run --rm test-cpu
```

The service uses CPU-only PyTorch 2.10, downloads/reuses the checksum-verified prepared artifacts, runs Normal + all 14 speculative methods in float32, and writes:

```text
cpu-reports/index.html
cpu-reports/report.html
cpu-reports/benchmark.json
```

Artifacts are cached in `cpu-artifacts/` and the Hugging Face model cache in `hf-cache/`.

Useful environment controls:

```text
CPU_THREADS=2
CPU_TOKENS=24
CPU_REPEATS=3
CPU_PROMPT_LIMIT=6
CPU_CALIBRATION_TOKENS=8
CPU_CALIBRATION_PROMPT_LIMIT=3
```

PowerShell smoke-sized example:

```powershell
$env:CPU_TOKENS="8"
$env:CPU_REPEATS="1"
$env:CPU_PROMPT_LIMIT="1"
docker compose --profile cpu run --rm test-cpu
```

See [`docs/cpu.md`](docs/cpu.md).

## GPU test with Docker Compose

Run the full GPU benchmark with:

```bash
docker compose --profile gpu run --rm test-gpu
```

Prerequisites:

- NVIDIA GPU + recent NVIDIA driver;
- Docker;
- NVIDIA Container Toolkit configured for Docker;
- Docker Compose v2 with GPU support.

The service does **not** silently fall back to CPU. It checks `torch.cuda.is_available()`, detected GPU(s), memory and compute capability, PyTorch/CUDA/cuDNN versions, and executes a real FP16 CUDA matrix multiplication before benchmarking.

GPU outputs:

```text
gpu-reports/index.html
gpu-reports/report.html
gpu-reports/benchmark.json
```

Artifacts are cached in `gpu-artifacts/`.

Default target dtype is FP16:

```text
LFM_GPU_DTYPE=float16
```

Supported values are `float16`, `bfloat16`, and `float32`.

The CUDA runtime moves the **LFM target/verifier and DFlash drafter** to GPU. Small MOBS/DSpark/Parareal routing remains CPU/NumPy so the selection logic stays aligned with the CPU implementation; host/device transfer cost is included in wall-clock measurements.

See [`docs/gpu.md`](docs/gpu.md).

## CPU vs GPU

Use the two matched surfaces independently:

```bash
docker compose --profile cpu run --rm test-cpu
docker compose --profile gpu run --rm test-gpu
```

CPU results live in `cpu-reports/`; GPU results live in `gpu-reports/`. Do not treat CPU and GPU rankings as one experiment. Compare absolute throughput only when prompts, tokens, repeats, artifacts, and benchmark settings are matched.

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
Dockerfile.cpu                              zero-install CPU image
Dockerfile.gpu                              zero-install CUDA image
docker-compose.yml                          test-cpu + test-gpu profiles
run_cpu.py                                  zero-install CPU runner
run_gpu.py                                  zero-install GPU runner + artifact downloader
src/dflash_mini_lab/lfm_runtime.py          CPU LFM2.5 reference runtime
src/dflash_mini_lab/lfm_gpu_runtime.py      CUDA target + DFlash drafter runtime
src/dflash_mini_lab/lfm_gpu_check.py        CUDA capability/matmul verification
src/dflash_mini_lab/lfm_gpu_benchmark.py    GPU All-14 benchmark entrypoint
src/dflash_mini_lab/lfm_all14_benchmark.py  canonical All-14 method/report logic
src/dflash_mini_lab/v13_minop.py            V13 MinOp
src/dflash_mini_lab/v14_simple_parareal.py  V14 scalar Parareal
docs/cpu.md                                 CPU local protocol
docs/gpu.md                                 GPU local protocol
.github/workflows/publish-artifacts.yml      persistent prepared-artifact release
.github/workflows/lfm-real-benchmark.yml     canonical CPU evidence workflow
```

## References

- Jian Chen, Yesheng Liang, Zhijian Liu. **DFlash: Block Diffusion for Flash Speculative Decoding.** arXiv:2602.06036, 2026.
- Official DFlash project: https://github.com/z-lab/dflash
- Vahid Tavakkoli et al. **Parareal Contribution to Speeding-Up the Solving of Nonlinear Ordinary Differential Equations on Parallel/Multi-Core Platforms for Sensing Systems.** The coarse/fine residual-correction principle motivates V12 and V14; these decoding variants are adaptations rather than claims of mathematical equivalence to the ODE solver.
- LiquidAI LFM2.5 model family.

## License

MIT.
