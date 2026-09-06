# DFlash Mini Lab

A reproducible **CPU speculative-decoding research lab centered on one canonical target: `LiquidAI/LFM2.5-350M-Base`**.

The active benchmark compares **12 speculative mechanisms** under one matched LFM2.5 protocol, plus normal target-only greedy decoding as the baseline. Every speculative output is verified by the same target model and must exactly match normal greedy output.

> [!IMPORTANT]
> This repository is a mechanism-level research/reference implementation. DFlash3 through DFlash12 are experimental lab variants and are not upstream official DFlash releases.

## Canonical result page

The GitHub Pages site is generated only from the unified LFM2.5 workflow:

**https://vtavakkoli.github.io/DFlash-Mini-Lab/**

The page includes:

- speedup vs normal decoding for every method;
- median tokens/second;
- draft acceptance;
- target-forward count and tokens/target-call;
- guidance-work diagnostics;
- exactness status;
- an interactive method explorer;
- an animated mechanism simulation for each speculative decoder;
- a machine-readable `benchmark.json` artifact.

Cross-model Qwen/EAGLE benchmark results are intentionally excluded from the canonical page and active CI evidence.

## The 12 speculative methods

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
| 09 | V9 DSpark-Lite | Low-rank Markov correction + survival gate | O(BK) |
| 10 | V10 Advanced Boltzmann | Sparse uncertainty routing | sparse O(BK) |
| 11 | V11 Boltzmann-Gated MOBS | Uncertainty-gated sparse MOBS | bounded O(BK) |
| 12 | **V12 PARAREAL** | **Parallel affine fine-minus-coarse residual correction** | **O(RBKD)** |

Normal autoregressive LFM2.5 is reported separately as method `00` and defines the 1.000× speed baseline.

## DFlash12-PARAREAL

V12 transfers the **coarse/fine residual-correction principle** of Parareal into speculative decoding while keeping the correction extremely small and block-parallel.

```text
DFlash coarse top-k field G
          │
          ▼
       q₀ = G
          │
     linear F-G
   residual round 1
          │
          ▼
         q₁
          │
     linear F-G
   residual round 2
          │
          ▼
         q₂
          │
          ▼
   speculative block
          │
          ▼
   LFM2.5 TARGET VERIFY
          │
          ▼
   exact greedy output
```

During preparation only, the frozen LFM2.5 target supplies the fine score field `F`. A closed-form ridge regression learns a compact approximation of the residual `F - q`. At inference, all retained candidate rows are corrected using vectorized linear algebra; no extra Transformer correction model is introduced.

See [`docs/version12-parareal.md`](docs/version12-parareal.md) for the full equations, feature contract, convergence diagnostics, limitations, and research interpretation.

## Reproduce the canonical study

The supported evidence path is the LFM2.5 Docker workflow.

```bash
docker build -f Dockerfile.lfm -t dflash-lfm25 .
```

Prepare the frozen DFlash backbone and auxiliary models, then fit V12:

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
  --seeds real_benchmarks/train_seeds.json \
  --top-k 8 \
  --correction-rounds 2
```

Run the complete comparison:

```bash
python -m dflash_mini_lab.lfm_all12_benchmark \
  --aux lfm-artifacts/lfm_aux.pt \
  --dspark lfm-artifacts/lfm_dspark.pt \
  --v12-model lfm-artifacts/v12_parareal.json \
  --prompts real_benchmarks/prompts.json \
  --calibration-prompts real_benchmarks/calibration_prompts.json \
  --output-dir lfm-reports \
  --tokens 24 \
  --repeats 3 \
  --prompt-limit 6 \
  --top-k 8 \
  --cpu-threads 2
```

Outputs:

```text
lfm-reports/index.html
lfm-reports/report.html
lfm-reports/benchmark.json
```

## Benchmark discipline

The unified workflow follows these rules:

1. **One target model:** all published performance evidence uses LFM2.5-350M-Base.
2. **Held-out benchmark prompts:** benchmark prompts are not used to fit V12.
3. **Separate calibration prompts:** ACT, V9, V10 and V11 are calibrated outside the benchmark workload.
4. **Rotated execution order:** method order changes across prompt/repeat combinations to reduce systematic first-run bias.
5. **Exactness gate:** every final speculative sequence is compared with normal greedy LFM output.
6. **No result cherry-picking:** the page is regenerated from the complete workflow artifact, including negative speedups if they occur.
7. **Matched runtime:** the methods share the same model, CPU-thread settings, prompt workload and target verifier.

See [`docs/reproducibility.md`](docs/reproducibility.md) for the complete protocol.

## Repository layout

```text
src/dflash_mini_lab/lfm_runtime.py          LFM2.5 target + compact DFlash auxiliaries
src/dflash_mini_lab/lfm_benchmark.py        DFlash through DFlash6 decoders
src/dflash_mini_lab/lfm_showcase.py         DFlash7-ACT
src/dflash_mini_lab/lfm_dspark.py           V9 DSpark-Lite
src/dflash_mini_lab/lfm_v10.py              V10 selector
src/dflash_mini_lab/v11_boltzmann_mobs.py   V11 selector
src/dflash_mini_lab/v12_parareal.py         V12 linear residual correction
src/dflash_mini_lab/v12_prepare.py          V12 LFM teacher-data preparation
src/dflash_mini_lab/lfm_all12_benchmark.py  canonical Normal + 12 benchmark and report
.github/workflows/lfm-real-benchmark.yml     sole active benchmark/evidence workflow
docs/algorithm.md                            method and complexity notes
docs/reproducibility.md                      LFM2.5 benchmark contract
docs/version12-parareal.md                   V12 design document
```

Historical experiment modules may remain in source for provenance and comparison, but they are not part of the active benchmark, CI evidence, or GitHub Pages result surface.

## References

- Jian Chen, Yesheng Liang, Zhijian Liu. **DFlash: Block Diffusion for Flash Speculative Decoding.** arXiv:2602.06036, 2026.
- Official DFlash project: https://github.com/z-lab/dflash
- Vahid Tavakkoli et al. **Parareal Contribution to Speeding-Up the Solving of Nonlinear Ordinary Differential Equations on Parallel/Multi-Core Platforms for Sensing Systems.** The coarse/fine residual-correction principle motivates V12; the decoding implementation is an adaptation rather than a claim of mathematical equivalence to the ODE solver.
- LiquidAI LFM2.5 model family.

## License

MIT.
