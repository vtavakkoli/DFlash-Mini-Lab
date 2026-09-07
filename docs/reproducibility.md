# LFM2.5 All-14 reproducibility guide

## Canonical scope

The active evidence surface is intentionally narrow:

> **one target model, one matched CPU protocol, Normal + 14 speculative mechanisms.**

Published benchmark evidence and GitHub Pages use only:

```text
LiquidAI/LFM2.5-350M-Base
```

Historical Qwen, EAGLE and tiny-model code may remain for provenance, but those runs are outside the active CI evidence surface.

## Active workflow

The sole benchmark workflow is:

```text
.github/workflows/lfm-real-benchmark.yml
```

It performs:

1. build the CPU LFM Docker image;
2. prepare the common frozen DFlash backbone/candidate vocabulary;
3. prepare V9 DSpark-Lite;
4. fit V12 full-field PARAREAL on training trajectories;
5. fit V14's three-coefficient fine-gap estimator on training trajectories;
6. calibrate ACT, V9, V10 and V11 on separate calibration prompts;
7. reuse the selected V10 configuration for V13 MinOp without extra V13 tuning;
8. run Normal + all 14 speculative methods on held-out benchmark prompts;
9. verify every final sequence against normal greedy LFM2.5;
10. write `benchmark.json` and the HTML report;
11. publish the LFM2.5-only Pages site on `main`.

## Data separation

### Training seeds

```text
real_benchmarks/train_seeds.json
```

Used for the common DFlash auxiliaries, V9, V12 and V14 fitting.

### Calibration prompts

```text
real_benchmarks/calibration_prompts.json
```

Used only to choose bounded inference policies for:

- DFlash7-ACT;
- V9 DSpark-Lite;
- V10 Advanced Boltzmann;
- V11 Boltzmann-Gated MOBS.

V13 deliberately reuses V10's selected configuration and is not separately tuned.

### Benchmark prompts

```text
real_benchmarks/prompts.json
```

Used only for final comparative timing and exactness.

Benchmark prompts must not be used to fit V12/V14 or select ACT/V9/V10/V11 policies.

## V12 artifact

```text
lfm-artifacts/v12_parareal.json
```

Default V12 settings:

```text
top_k              = 8
correction_rounds  = 2
damping            = 0.75
ridge               = 0.001
```

The artifact contains linear residual parameters, normalization/configuration and train/holdout convergence diagnostics. It does not contain target weights.

## V14 artifact

```text
lfm-artifacts/v14_simple_parareal.json
```

Default canonical V14 settings:

```text
ridge             = 0.001
damping           = 1.0
switch_threshold  = 0.0
max_corrections   = 1
```

The artifact contains only:

- three fine-gap estimator coefficients;
- configuration;
- model/block metadata;
- train and holdout fine-gap diagnostics.

The target fine gap exists only during preparation.

## Canonical benchmark command

After preparing the common/V9/V12/V14 artifacts:

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
  --calibration-tokens 8 \
  --calibration-prompt-limit 3 \
  --top-k 8 \
  --cpu-threads 2 \
  --dtype float32
```

## Methods in the run

The output has 15 rows:

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
13 minop_v13
14 simple_parareal_v14
```

## Fair timing discipline

The unified benchmark:

- warms the target and drafter before measured work;
- uses the same target, candidate backbone, prompts, output length and CPU settings;
- rotates method order across prompt/repeat combinations;
- reports median tokens/second and median latency;
- records target, draft and selection/correction time;
- records target-forward count and tokens per target pass;
- preserves negative results.

V13's `torch.topk(2)` is included in its measured draft time. This is important: the optimization is not treated as free work.

V14's scalar estimator work is included in selection time and reported separately as estimator operations.

## Exactness gate

A speculative result is valid only when:

```text
all_exact == true
```

Exactness means the complete generated sequence equals normal greedy LFM2.5 for the same prompt and output length. The workflow fails if any of the 15 paths is not exact.

## Speed claims

Only end-to-end wall-clock throughput from the same run supports a speed claim.

The following are diagnostics, not speed proofs by themselves:

- draft acceptance;
- tokens per target call;
- target-forward count;
- guidance-operation count;
- V12 teacher-space error;
- V14 fine-gap MAE/sign accuracy;
- V13 routed-position count.

## Outputs and provenance

Canonical outputs:

```text
lfm-reports/index.html
lfm-reports/report.html
lfm-reports/benchmark.json
```

For publication-quality archiving retain:

- repository commit SHA;
- `benchmark.json`;
- `v12_parareal.json`;
- `v14_simple_parareal.json`;
- target model ID/revision;
- auxiliary artifact manifest;
- training/calibration/benchmark prompt files;
- Docker definition and CLI arguments;
- GitHub Actions run ID;
- CPU/OS/container metadata.

See [`algorithm.md`](algorithm.md), [`version12-parareal.md`](version12-parareal.md), and [`version13-14.md`](version13-14.md).
