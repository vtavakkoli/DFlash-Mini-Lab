# DFlash Mini Lab

A reproducible **CPU speculative-decoding research lab centered on one canonical target: `LiquidAI/LFM2.5-350M-Base`**.

The active benchmark compares **14 speculative mechanisms** under one matched LFM2.5 protocol, plus normal target-only greedy decoding as the baseline. Every speculative output is verified by the same target model and must exactly match normal greedy output.

> [!IMPORTANT]
> This repository is a mechanism-level research/reference implementation. DFlash3 through DFlash14 are experimental lab variants and are not upstream official DFlash releases.

## Canonical result page

**https://vtavakkoli.github.io/DFlash-Mini-Lab/**

The GitHub Pages site is generated from the unified LFM2.5 workflow and reports speedup, tokens/second, acceptance, target-forward count, tokens/target-call, selector/correction work, exactness, and an animated mechanism explorer. The machine-readable source of truth is `benchmark.json`.

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

Normal autoregressive LFM2.5 is method `00` and defines the exact reference and `1.000×` speed baseline.

## V13 MinOp — simplified and optimized V10

The previous LFM2.5 study showed that V10 improved draft acceptance and target-pass efficiency but paid a small selector cost. V13 keeps the useful V10 decision policy while removing avoidable data movement and search work.

```text
context
  │
  ▼
DFlash drafter
  │
  ├─ full logits stay in Torch
  │
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

V13 deliberately **reuses the configuration selected for V10**. It is not separately tuned. This makes the V10→V13 comparison primarily an implementation/operation-cost experiment.

## V14 Simple PARAREAL — simple estimator for the fine model

V12 learns a residual over the complete `B × K` score field. V14 asks whether the Parareal idea can be compressed to the smallest useful state: the signed gap between the draft top-1 and top-2 candidates.

For each block position:

```text
G = draft_top1_logit - draft_top2_logit
F = target_logit(draft_top1) - target_logit(draft_top2)   # preparation only
```

V14 fits only three coefficients:

```text
F_hat = a + b·G + c·normalized_position
```

and performs one Parareal-style update:

```text
gap_1 = G + damping · (F_hat - G)
```

At inference, at most one position may switch from draft top-1 to draft top-2. The estimator is ordinary closed-form ridge regression; there is no neural correction model and no target-model call beyond the normal authoritative verifier.

The artifact `lfm-artifacts/v14_simple_parareal.json` stores the three coefficients, configuration, and train/holdout diagnostics only. It does not redistribute target weights.

See [`docs/version13-14.md`](docs/version13-14.md) for the design and interpretation rules.

## Reproduce the canonical study

Build the CPU image:

```bash
docker build -f Dockerfile.lfm -t dflash-lfm25 .
```

Prepare the common backbone, V9, V12 and V14 artifacts:

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

1. **One target model:** all active performance evidence uses LFM2.5-350M-Base.
2. **Held-out benchmark prompts:** benchmark prompts are not used to fit V12 or V14.
3. **Separate calibration prompts:** ACT, V9, V10 and V11 are calibrated outside the benchmark workload; V13 reuses the selected V10 policy.
4. **Rotated execution order:** method order changes across prompt/repeat combinations.
5. **Exactness gate:** all 15 paths—Normal + 14 speculative methods—must reproduce normal greedy LFM output.
6. **No result cherry-picking:** negative speedups are retained.
7. **Matched runtime:** methods share the model, CPU settings, workload and verifier.

See [`docs/reproducibility.md`](docs/reproducibility.md) for the complete protocol.

## Repository layout

```text
src/dflash_mini_lab/lfm_runtime.py            LFM2.5 target + compact DFlash auxiliaries
src/dflash_mini_lab/lfm_benchmark.py          DFlash through DFlash6
src/dflash_mini_lab/lfm_showcase.py           DFlash7-ACT
src/dflash_mini_lab/lfm_dspark.py             V9 DSpark-Lite
src/dflash_mini_lab/lfm_v10.py                V10 selector
src/dflash_mini_lab/v11_boltzmann_mobs.py     V11 selector
src/dflash_mini_lab/v12_parareal.py           V12 full-field linear residual correction
src/dflash_mini_lab/v13_minop.py              V13 fused top-2 MinOp decoder
src/dflash_mini_lab/v14_simple_parareal.py    V14 scalar Parareal decoder
src/dflash_mini_lab/v14_prepare.py             V14 fine-gap estimator fitting
src/dflash_mini_lab/lfm_all14_benchmark.py    canonical Normal + 14 benchmark/report
.github/workflows/lfm-real-benchmark.yml       sole active benchmark/evidence workflow
```

## References

- Jian Chen, Yesheng Liang, Zhijian Liu. **DFlash: Block Diffusion for Flash Speculative Decoding.** arXiv:2602.06036, 2026.
- Official DFlash project: https://github.com/z-lab/dflash
- Vahid Tavakkoli et al. **Parareal Contribution to Speeding-Up the Solving of Nonlinear Ordinary Differential Equations on Parallel/Multi-Core Platforms for Sensing Systems.** The coarse/fine residual-correction principle motivates V12 and V14; these decoding variants are adaptations rather than claims of mathematical equivalence to the ODE solver.
- LiquidAI LFM2.5 model family.

## License

MIT.
