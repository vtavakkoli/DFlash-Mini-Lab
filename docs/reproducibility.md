# Reproducibility guide

## Scope

This repository contains two distinct reproducibility surfaces:

1. the compact/tiny CPU reference benchmark, designed for fast mechanism-level regression tests; and
2. real-model LFM/Qwen experiments, including **DFlash12-PARAREAL**, which require the corresponding frozen target and auxiliary artifacts.

Results from one surface must not be presented as measurements from the other.

## Tiny runtime contract

The default Docker image is CPU-only and intentionally keeps the runtime small. The model-builder stage creates the compact target/drafter artifacts deterministically and the final runtime executes the NumPy reference implementation.

Recommended single-process settings:

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

The tiny path is intended for algorithm mechanics, regression testing, exactness checks, and controlled relative comparisons. Its absolute throughput must not be compared directly with production inference engines.

## DFlash12-PARAREAL reproducibility contract

DFlash12 separates **preparation** from **inference/benchmarking**.

During preparation, a frozen LFM target is used to generate greedy teacher trajectories and fine logits. A compact ridge-regression artifact is then written to:

```text
lfm-artifacts/v12_parareal.json
```

The JSON artifact contains only the linear correction parameters, feature-normalization statistics, configuration, and convergence metadata. It does not redistribute the target-model weights.

Default V12 numerical settings are:

```text
top_k              = 8
correction_rounds  = 2
damping            = 0.75
ridge               = 0.001
residual_clip       = 6.0
interpolation       = [0.0, 0.5, 0.75]
random seed         = 23
```

The complete fitted configuration is stored in the artifact and should be reported with any benchmark result.

## Prepare V12 deterministically

Use a fixed training-seed file and fixed CPU-thread count:

```bash
python -m dflash_mini_lab.v12_prepare \
  --aux lfm-artifacts/lfm_aux.pt \
  --seeds real_benchmarks/train_seeds.json \
  --output lfm-artifacts/v12_parareal.json \
  --max-seed-count 24 \
  --generation-tokens 24 \
  --top-k 8 \
  --correction-rounds 2 \
  --damping 0.75 \
  --ridge 0.001 \
  --cpu-threads 2 \
  --seed 23
```

The preparation process:

1. fixes Python, NumPy, and Torch RNG seeds;
2. generates greedy target continuations;
3. performs one full causal teacher pass over each completed trajectory and reuses those logits for all legal block windows;
4. constructs DFlash top-k coarse score fields and target fine score fields on identical candidate IDs;
5. fits the residual model with closed-form ridge regression;
6. reserves an internal holdout partition;
7. stores train and holdout convergence diagnostics in the JSON artifact.

Because the model is fit with a deterministic linear solve, repeated runs with the same software stack, target/auxiliary artifacts, seed data, thread settings, and numerical libraries should produce equivalent fitted parameters up to the numerical behavior of the underlying BLAS/LAPACK implementation.

## Train/holdout separation

The preparation holdout is used only to diagnose whether the learned residual contracts teacher-space error. It is not used to choose target tokens during inference.

The reported diagnostics are:

- teacher-space MSE by correction round;
- `log(MSE)` by correction round;
- contraction ratio `E_(k+1) / E_k`;
- fine-model top-1 agreement by correction round.

A synthetic unit-test convergence result is not evidence of real-model convergence. Real-model claims must come from the holdout metrics stored in a newly prepared V12 artifact.

## Benchmark V12

Use a prompt file that is separate from the preparation seeds:

```bash
python -m dflash_mini_lab.v12_benchmark \
  --aux lfm-artifacts/lfm_aux.pt \
  --dspark lfm-artifacts/lfm_dspark.pt \
  --v12-model lfm-artifacts/v12_parareal.json \
  --prompts real_benchmarks/test_prompts.json \
  --output-dir v12-reports \
  --tokens 24 \
  --repeats 2 \
  --cpu-threads 2
```

The benchmark writes:

```text
v12-reports/v12_benchmark.json
v12-reports/v12_benchmark.md
```

The JSON file is the machine-readable source of truth. The Markdown report is a human-readable summary.

## What a V12 benchmark records

For Normal LFM, DFlash, V11, and V12, the benchmark records:

- end-to-end generated tokens/sec;
- end-to-end latency;
- target time;
- draft time;
- selection/correction time;
- target forward-pass count;
- draft acceptance rate;
- generated tokens per target pass;
- exact-output comparison with normal greedy decoding.

V12 additionally records linear candidate-score work and the average correction-update RMS.

## Exactness requirement

Speculative proposals are never accepted without target verification. A benchmark result should be treated as valid only when `all_exact` is true for the method under discussion.

Exactness means that the complete generated sequence matches the normal target-only greedy sequence for the same prompt and output length. Teacher-space convergence metrics do not replace this check.

## Fair timing protocol

For comparative measurements:

- use the same machine and power/performance mode;
- keep CPU-thread settings fixed across methods;
- use the same target and auxiliary artifacts;
- use the same prompt order and output length;
- avoid mixing cold-start/model-load time with decode timing unless explicitly studying startup;
- retain multiple repeats and report the aggregation rule;
- preserve negative results rather than retuning until a preferred ranking appears.

The V12 benchmark rotates method order across prompt/repeat pairs to reduce systematic ordering bias.

## Artifact provenance

For a publication-quality experiment, archive or record alongside the result:

- repository commit SHA;
- V12 JSON artifact;
- target model ID/revision if available;
- auxiliary artifact identity;
- training seed file and test prompt file;
- exact CLI command;
- Python and package versions;
- CPU model and thread settings;
- operating-system/container information.

If artifacts are regenerated, treat them as a new experimental condition rather than assuming equivalence with an earlier run.

## Unit tests

The V12 unit tests validate:

- geometric contraction on a controlled affine residual problem;
- deterministic, shape-preserving parallel correction;
- lossless JSON save/load of the fitted linear model.

Run:

```bash
pytest -q tests/test_v12_parareal.py
```

or the complete repository suite:

```bash
pytest -q
```

## Interpretation discipline

A lower correction error, higher acceptance rate, or fewer target calls does not by itself establish an end-to-end speedup. Only measured wall-clock throughput under the same workload supports a speed claim.

Likewise, DFlash12 is **Parareal-inspired**: it borrows the coarse/fine residual-correction idea but operates on speculative logit fields with a learned linear surrogate. The repository does not claim equivalence to the classical nonlinear-ODE Parareal algorithm.

See [`version12-parareal.md`](version12-parareal.md) for the complete V12 design and [`algorithm.md`](algorithm.md) for the broader method comparison.
