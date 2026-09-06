# DFlash Mini Lab

A reproducible, CPU-oriented research lab for speculative decoding mechanisms. The repository contains compact educational implementations, real-model LFM2.5 and Qwen paths, exact greedy verification, benchmark tooling, and versioned experiments from DFlash-style block drafting through **DFlash12-PARAREAL**.

> [!IMPORTANT]
> This repository is a mechanism-level research/reference implementation. It is **not** the official DFlash, DFlash2, EAGLE, vLLM, SGLang, or vendor runtime. Experimental versions introduced here are named for this lab and should not be presented as upstream algorithms.

## Current research focus: DFlash12-PARAREAL

**DFlash12-PARAREAL** transfers the coarse/fine residual-correction idea of Parareal into speculative decoding while keeping the block correction parallel and extremely small.

The mapping is:

- **coarse propagator `G`**: the existing DFlash top-k block-logit field;
- **fine propagator `F` during preparation only**: frozen target-model teacher logits on greedy trajectories;
- **residual surrogate**: a closed-form ridge/least-squares model trained to predict `F - q`;
- **inference correction**: repeated affine residual updates over every block-position/candidate pair in parallel;
- **authority**: the unchanged target verifier accepts the matching prefix and corrects the first mismatch.

The correction state is a continuous top-k logit field rather than token IDs. For correction round `k`:

```text
q_0 = center(G)
q_(k+1) = center(q_k + damping * R_linear(q_k, G, features))
```

`R_linear` is ordinary ridge regression. The default model has only eight standardized features plus an intercept. It is evaluated with vectorized NumPy over the complete `B × K` candidate field, so V12 adds **no neural correction forward pass** and introduces no position-by-position recurrence.

### Why Parareal here?

Classical Parareal alternates a cheap coarse approximation with a correction based on the difference between fine and coarse propagation. V12 keeps that residual-correction principle but adapts it to a speculative block:

```text
Frozen target teacher F  ───────┐
                                │  preparation only
DFlash coarse field G ──────────┼──> fit linear residual F-G
                                │
                                ▼
                         v12_parareal.json
                                │
                                ▼
DFlash block logits ──> q0 ──> q1 ──> q2 ──> proposal
                         all B×K candidates corrected in parallel
                                │
                                ▼
                         TARGET VERIFY
                                │
                                ▼
                       exact greedy output
```

The implementation records teacher-space mean-squared error, `log(MSE)`, contraction ratios, and fine top-1 agreement by correction round. These are preparation/holdout diagnostics; they are not used as an oracle at inference time.

## DFlash12 feature contract

For every retained candidate, the linear model receives:

1. current centered candidate score;
2. original coarse centered score;
3. candidate rank within the coarse top-k set;
4. normalized block position;
5. coarse top-1/top-2 margin;
6. candidate-embedding similarity to the last prefix token;
7. candidate-embedding similarity to the prefix-mean embedding;
8. current-score × block-position interaction.

Training augments each teacher block with intermediate points between the coarse and fine fields. This teaches the same affine residual model to correct not only `q0`, but also partially corrected states.

## Version map

| Version | Experiment | Main idea |
|---|---|---|
| DFlash | Parallel block drafting | Predict a future block, then verify with target |
| DFlash2-style | Dynamic-programming path selection | Top-k candidates with predecessor-conditioned scoring |
| DFlash3-MOBS | Middle-out bidirectional selection | O(BK) local path construction |
| DFlash4-JUMP-MOBS | Sparse jump anchors | Predict selected future positions, fill gaps |
| DFlash5-FUSED-JUMP-MOBS | Fused sparse residual | Reuse drafter hidden states, remove separate jump forward |
| DFlash6 | Boltzmann / BMOBS | Deterministic training-free exploration |
| DFlash7-ACT | Adaptive speculation | Shorten uncertain verifier suffixes on cached Qwen |
| V8 | EAGLE3-oriented experiment | Separate EAGLE3 comparison path |
| V9 DSpark-Lite | Low-rank Markov correction | Frozen DFlash backbone + confidence head |
| V10 Advanced Boltzmann | Cost-aware candidate exploration | Training-free selection refinements |
| V11 Boltzmann-Gated MOBS | Sparse uncertainty routing | Send only uncertain slots through MOBS |
| **V12 PARAREAL** | **Parallel linear residual correction** | **Least-squares approximation of fine-minus-coarse logit residual** |

## DFlash12 preparation

V12 is trained offline from frozen teacher trajectories. The target model is used only to build the regression data; target weights are not written into the V12 JSON artifact.

```bash
python -m dflash_mini_lab.v12_prepare \
  --aux lfm-artifacts/lfm_aux.pt \
  --seeds real_benchmarks/train_seeds.json \
  --output lfm-artifacts/v12_parareal.json \
  --top-k 8 \
  --correction-rounds 2 \
  --damping 0.75 \
  --ridge 0.001
```

The preparation command:

- generates deterministic greedy teacher trajectories;
- performs one causal teacher pass per completed trajectory for reusable fine logits;
- collects DFlash top-k coarse scores;
- fits the linear `F-G` residual with closed-form ridge regression;
- uses an internal holdout split;
- stores train/holdout convergence diagnostics in the JSON artifact.

## DFlash12 benchmark

The real LFM comparison keeps final target verification identical across speculative methods and checks every generated sequence against normal target-only greedy decoding.

```bash
python -m dflash_mini_lab.v12_benchmark \
  --aux lfm-artifacts/lfm_aux.pt \
  --dspark lfm-artifacts/lfm_dspark.pt \
  --v12-model lfm-artifacts/v12_parareal.json \
  --prompts real_benchmarks/test_prompts.json \
  --output-dir v12-reports \
  --tokens 24 \
  --repeats 2
```

Outputs:

```text
v12-reports/v12_benchmark.json
v12-reports/v12_benchmark.md
```

The benchmark compares:

- Normal LFM greedy decoding;
- plain DFlash;
- V11 Boltzmann-Gated MOBS;
- V12 PARAREAL Linear.

It records end-to-end tok/s, target/draft/selection time, target-forward count, acceptance, tokens per target pass, V12 linear-candidate work, and exactness.

## Synthetic convergence regression test

The unit test intentionally uses a synthetic affine fine/coarse relationship where the residual is representable by the V12 feature model. This verifies the numerical mechanism independently from real-model quality. In the development run used to introduce V12, the synthetic holdout MSE behaved approximately as:

```text
round 0: 0.1863
round 1: 0.0123
round 2: 0.00084
```

That result validates the implementation's geometric-convergence behavior on a controlled affine problem. It is **not** an LFM throughput or quality result.

Run tests with:

```bash
pytest -q
```

## Qwen3-0.6B path: DFlash7-ACT

The cached Qwen path remains separate from V12. It uses `DynamicCache`, a learned hidden-fusion block drafter, one known greedy anchor, speculative suffix verification, and cache cropping after rejection. DFlash7-ACT uses top-1/top-2 draft margin to suppress unprofitable suffixes.

The preserved first Qwen result is intentionally negative end-to-end: the small drafter did not beat normal cached Qwen on the measured CPU workload. The repository keeps this result because acceptance and speculative sophistication do not automatically imply wall-clock speedup.

## Tiny reference path

The original tiny Docker benchmark remains useful for mechanism-level regression tests and visualization. Its target intentionally recomputes the visible sequence and should not be compared with production serving numbers from llama.cpp, vLLM, SGLang, or vendor runtimes.

```bash
docker build -t dflash-mini-lab .
docker run --rm \
  -e CPU_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 \
  -e OMP_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  -v "$PWD/reports:/app/reports" \
  dflash-mini-lab
```

## Repository layout

```text
src/dflash_mini_lab/runtime.py             tiny NumPy reference runtime
src/dflash_mini_lab/decoding.py            tiny speculative methods + verification
src/dflash_mini_lab/lfm_runtime.py          real LFM reference runtime
src/dflash_mini_lab/lfm_dspark.py           V9 DSpark-Lite runtime/training
src/dflash_mini_lab/lfm_v10.py              V10 configuration/selection
src/dflash_mini_lab/v11_boltzmann_mobs.py   V11 uncertainty-gated MOBS
src/dflash_mini_lab/v11_benchmark.py         V11 real-model benchmark
src/dflash_mini_lab/v12_parareal.py          V12 regression + parallel correction core
src/dflash_mini_lab/v12_prepare.py           V12 teacher-data and least-squares preparation
src/dflash_mini_lab/v12_benchmark.py         V12 exactness/performance benchmark
src/dflash_mini_lab/qwen_runtime.py          cached Qwen verifier + rollback
src/dflash_mini_lab/qwen_benchmark.py        Normal / DFlash / DFlash7 benchmark
tests/test_v12_parareal.py                   V12 convergence/determinism/artifact tests
docs/algorithm.md                            algorithm and complexity notes
docs/reproducibility.md                      reproducibility contract
docs/version12-parareal.md                   V12 design, equations, claims and limitations
```

## Documentation

- [`docs/algorithm.md`](docs/algorithm.md) — algorithm families, equations, complexity and exactness.
- [`docs/reproducibility.md`](docs/reproducibility.md) — deterministic settings, artifact contract and benchmark interpretation.
- [`docs/version8-eagle3.md`](docs/version8-eagle3.md) — V8/EAGLE3 experiment notes.
- [`docs/version12-parareal.md`](docs/version12-parareal.md) — complete V12 design and research protocol.

## References

- J. Chen, Y. Liang, Z. Liu. **DFlash: Block Diffusion for Flash Speculative Decoding.** arXiv:2602.06036, 2026.
- Official DFlash project: https://github.com/z-lab/dflash
- V. Tavakkoli et al. **Parareal Contribution to Speeding-Up the Solving of Nonlinear Ordinary Differential Equations on Parallel/Multi-Core Platforms for Sensing Systems.** The Parareal coarse/fine residual-correction principle motivates V12; the decoding implementation is an adaptation, not a claim of mathematical equivalence to the ODE solver.
- vLLM Speculators documentation.
- Qwen3 model family documentation.

## License

MIT.
