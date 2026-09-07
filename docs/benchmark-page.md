# LFM2.5 All-14 benchmark page

The GitHub Pages site is the human-readable view of the canonical LFM2.5 benchmark artifact.

## Purpose

The page answers five questions without mixing models or benchmark protocols:

1. Which speculative method is fastest relative to normal LFM2.5 greedy decoding?
2. How much draft work is accepted before target correction?
3. How many target forwards are required for the same generated-token workload?
4. How much selector/correction work does each method add?
5. Do V13 MinOp and V14 Simple PARAREAL improve the cost/benefit tradeoff of V10 and V12?

## Source of truth

The page is generated from the same in-memory result object written to:

```text
lfm-reports/benchmark.json
```

No performance number is manually embedded. Every canonical run regenerates `benchmark.json`, `report.html`, and `index.html`.

## Page sections

### Hero and exactness status

Shows the target model, fastest speculative method, best measured speedup, and exactness across all **15 paths**: Normal + 14 speculative methods.

### Speedup versus normal decoding

Displays median generated tokens/second normalized by normal LFM2.5. Normal is always `1.000×`.

### V13 comparison card

Compares V13 MinOp directly with V10 Advanced Boltzmann. V13 reuses the V10 selected policy but moves top-2 extraction into Torch and hard-limits routing to one slot.

The card exposes:

- V13 speedup;
- selector time;
- guidance work;
- V10 reference speedup and selector time.

### V14 comparison card

Compares V14 Simple PARAREAL with V12 full-field PARAREAL. It shows:

- V14 speedup;
- guidance reduction relative to V12;
- V14 selector time;
- V14 holdout fine-gap MAE and sign accuracy.

### Complete table

For all 15 paths the table reports:

- median tokens/second;
- speedup vs normal;
- draft acceptance;
- target-forward count;
- tokens per target call;
- selection/correction time;
- guidance-work diagnostic;
- exactness.

### Mechanism simulation

The interactive explorer animates the four conceptual stages of every speculative method. For the new methods:

```text
V13: FUSED TOP-2 -> ONE SLOT -> VERIFY -> COMMIT
V14: COARSE GAP -> F-HAT -> VERIFY -> COMMIT
```

The animation is explanatory. It is not itself a benchmark measurement.

## Publication rule

Only the LFM2.5 report and its JSON artifact are copied into the Pages artifact. Historical cross-model artifacts are not fetched or published by the active workflow.

## Exactness rule

The site is published only after every method has `all_exact == true`. A failed exactness check fails the evidence job before deployment.
