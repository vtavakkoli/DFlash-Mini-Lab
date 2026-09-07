# LFM2.5 All-12 benchmark page

The GitHub Pages site is the human-readable view of the canonical LFM2.5 benchmark artifact.

## Purpose

The page answers four questions without mixing models or benchmark protocols:

1. Which speculative method is fastest relative to normal LFM2.5 greedy decoding?
2. How much draft work is accepted before target correction?
3. How many target forwards are required for the same generated-token workload?
4. What mechanism distinguishes each of the twelve methods?

## Source of truth

The page is generated from the same in-memory result object written to:

```text
lfm-reports/benchmark.json
```

No performance number is manually embedded in the site. On each canonical run the workflow regenerates both `benchmark.json` and `report.html`.

## Page sections

### Hero and run status

Shows:

- target model;
- fastest speculative method;
- best measured speedup;
- exactness status across Normal + all 12 speculative paths.

### Speedup chart

Displays median tokens/second normalized by the normal LFM2.5 baseline. The normal baseline is always `1.000×`.

### Efficiency frontier

Plots draft acceptance against speedup to make it visually clear that higher acceptance does not necessarily imply higher throughput.

### Complete table

For every method the table reports:

- median tokens/second;
- speedup vs normal;
- draft acceptance;
- mean target-forward count;
- tokens per target call;
- guidance-work diagnostic;
- exactness.

### Method evolution

Twelve selectable cards summarize the mechanism progression from plain DFlash block drafting through V12 PARAREAL.

### Mechanism simulation

The explorer animates four conceptual phases for the selected decoder, such as:

```text
DRAFT -> GUIDANCE -> VERIFY -> COMMIT
```

or for V12:

```text
COARSE G -> F-G x2 -> VERIFY -> COMMIT
```

The simulation is explicitly explanatory. It uses aggregate measured acceptance/routing information where available, but it is not itself a benchmark measurement.

## Publication rule

Only the LFM2.5 report and its JSON artifact are copied into the Pages artifact. Historical cross-model artifacts are not fetched, linked, or published by the active workflow.

## Exactness rule

The site is published only after the workflow verifies that every method has `all_exact == true`. A failed exactness check fails the evidence job before deployment.
