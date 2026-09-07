# LFM2.5 All-12 reproducibility guide

## Canonical scope

The active evidence surface of this repository is deliberately narrow:

> **one target model, one matched CPU protocol, normal decoding plus 12 speculative mechanisms.**

Published benchmark evidence and GitHub Pages use only:

```text
LiquidAI/LFM2.5-350M-Base
```

Historical experiment code may remain in the repository, but Qwen, EAGLE and tiny-model runs are not part of the active CI result surface.

## Active workflow

The sole benchmark workflow is:

```text
.github/workflows/lfm-real-benchmark.yml
```

It performs the complete experiment from preparation through report generation:

1. build the CPU LFM Docker image;
2. prepare the frozen LFM2.5 DFlash backbone and candidate vocabulary;
3. train the V9 DSpark-Lite heads;
4. fit the V12 PARAREAL linear residual model on training trajectories;
5. calibrate ACT, V9, V10 and V11 on calibration prompts;
6. run Normal + all 12 speculative methods on held-out benchmark prompts;
7. verify every final output against normal greedy LFM2.5;
8. write `benchmark.json` and the professional HTML report;
9. publish the LFM2.5-only GitHub Pages site on `main`.

## Fixed target and runtime

The canonical target is loaded in float32 CPU mode with two CPU threads in the GitHub Actions study. The workflow also pins the relevant package versions through the Docker build.

The benchmark methods share:

- target model and target revision available through the same Hugging Face cache;
- candidate vocabulary and DFlash backbone;
- benchmark prompts;
- output length;
- CPU thread settings;
- exact target-verification logic.

Do not compare absolute throughput from this reference runtime with a production serving engine unless the serving backend and measurement protocol are also matched.

## Data separation

The repository keeps three logically separate data roles.

### Training seeds

```text
real_benchmarks/train_seeds.json
```

Used to prepare the DFlash auxiliaries, V9 heads, and V12 regression artifact.

### Calibration prompts

```text
real_benchmarks/calibration_prompts.json
```

Used only to select bounded inference policies for methods that require calibration:

- DFlash7-ACT margin threshold;
- V9 DSpark survival floor;
- V10 bounded Boltzmann configuration;
- V11 uncertainty-gated MOBS configuration.

### Benchmark prompts

```text
real_benchmarks/prompts.json
```

Used only for the final comparative timing/exactness study.

Benchmark prompts must not be used to fit V12 or select the ACT/V9/V10/V11 policy.

## V12 preparation

Default canonical V12 settings:

```text
top_k              = 8
correction_rounds  = 2
damping            = 0.75
ridge               = 0.001
residual_clip       = 6.0
interpolation       = [0.0, 0.5, 0.75]
```

The V12 artifact is:

```text
lfm-artifacts/v12_parareal.json
```

It contains the regression coefficients, feature-normalization statistics, configuration, and train/holdout convergence diagnostics. It does not contain LFM target weights.

Preparation uses frozen greedy LFM teacher trajectories. The fine score field `F` exists only during preparation; inference uses the fitted linear residual surrogate.

## Canonical benchmark command

After preparation:

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
  --calibration-tokens 8 \
  --calibration-prompt-limit 3 \
  --top-k 8 \
  --cpu-threads 2 \
  --dtype float32
```

## Methods in the run

The output contains 13 rows:

```text
00 normal
01 dflash
02 dflash2
03 dflash3_mobs
04 dflash4_jump_mobs
05 dflash5_fused_jump_mobs
06 dflash6_boltzmann
07 dflash6_bmobs
08 dflash7_act
09 dspark_v9
10 boltzmann_v10
11 boltzmann_gated_mobs_v11
12 parareal_linear_v12
```

Normal defines the exact reference and 1.000× baseline; the other twelve are the speculative methods.

## Timing discipline

The unified benchmark:

- warms the target and drafter before measured work;
- computes a normal greedy reference for each prompt outside speculative exactness decisions;
- times each decoder end to end for generation work;
- rotates method order across prompt/repeat combinations;
- reports median tokens/second and median latency;
- records target, draft and selector/correction time;
- records target-forward counts and tokens per target pass;
- preserves every method result rather than publishing only the winner.

Method-order rotation reduces systematic first-method bias but does not eliminate all hosted-runner variance. Repeated runs should be used for publication-quality confidence intervals.

## Exactness gate

A speculative result is valid only when:

```text
all_exact == true
```

for the method.

Exactness means the full generated token sequence equals normal greedy LFM2.5 for the same prompt and output length. Acceptance rate, teacher-space convergence, or lower target-call count never substitutes for this check.

The workflow fails if any of the 13 paths is not exact.

## Speed claims

A speed claim requires measured wall-clock throughput from the same run.

The following are useful diagnostics but are not speed proofs by themselves:

- draft acceptance;
- tokens per target call;
- guidance-operation count;
- V12 teacher-space MSE;
- V12 contraction ratio;
- number of trimmed speculative tokens.

Negative speedups are retained in `benchmark.json` and on the page.

## Professional report contract

The generated report includes:

- a speedup-vs-normal chart;
- acceptance/speed efficiency view;
- complete exactness-gated table;
- method evolution cards;
- an interactive method explorer;
- an animated mechanism simulation for every speculative method;
- protocol and provenance notes.

The animation is explanatory. It uses the measured aggregate profile to visualize the mechanism, while `benchmark.json` remains the machine-readable source of truth.

Outputs:

```text
lfm-reports/index.html
lfm-reports/report.html
lfm-reports/benchmark.json
```

## Artifact provenance

For publication-quality archiving, retain:

- repository commit SHA;
- `benchmark.json`;
- `v12_parareal.json`;
- LFM target model ID/revision;
- auxiliary artifact manifest;
- training/calibration/benchmark prompt files;
- Docker image definition;
- exact CLI arguments;
- GitHub Actions run ID;
- CPU/OS/container metadata where available.

Regenerating a learned auxiliary or V12 artifact constitutes a new experimental condition.

## Interpretation of V12

DFlash12 is **Parareal-inspired**. It borrows the coarse/fine residual-correction principle and adapts it to a speculative top-k logit field using a learned linear surrogate. It is not claimed to be mathematically equivalent to the classical nonlinear-ODE Parareal solver.

See [`version12-parareal.md`](version12-parareal.md) for the complete V12 design and [`algorithm.md`](algorithm.md) for all 12 methods.
