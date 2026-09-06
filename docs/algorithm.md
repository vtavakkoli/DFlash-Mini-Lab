# LFM2.5 All-12 algorithm notes

The canonical benchmark in this repository evaluates **12 speculative-decoding mechanisms against one target model: `LiquidAI/LFM2.5-350M-Base`**. Normal target-only greedy decoding is method `00` and defines both the exact reference output and the `1.000×` throughput baseline.

All speculative methods are approximate until the same LFM2.5 target verifier accepts the matching prefix. Complete generated sequences are compared with normal greedy decoding.

> DFlash3 through DFlash12 are experimental lab variants. Their names describe this repository's research sequence and are not upstream official DFlash release numbers.

## Method 00 — Normal autoregressive

One target step produces one greedy token. No draft model or candidate selector is used.

## Method 01 — DFlash

A compact non-causal drafter predicts the full future block in parallel. The target verifies the proposed block and accepts its matching prefix.

**Guidance:** parallel draft argmax.  
**Cost model:** draft forward + target verification.

## Method 02 — DFlash2-style path selection

Retain top-`K` candidates at every future position and use predecessor-conditioned transition scores with dynamic programming.

```text
K + (B - 1) K²
```

**Guidance complexity:** `O(BK²)`.

## Method 03 — DFlash3-MOBS

MOBS chooses a central anchor and expands left/right, scoring only `K` candidates against already selected neighbors.

**Guidance complexity:** `O(BK)`.

## Method 04 — DFlash4-JUMP-MOBS

A separate jump head predicts sparse future anchors. MOBS-style local scoring fills the remaining gaps.

**Guidance complexity:** approximately `O(BK + JK)`, plus a separate jump-head forward pass.

## Method 05 — DFlash5-FUSED-JUMP

Reuse the DFlash hidden state to create sparse residual anchors, removing DFlash4's extra jump forward pass.

**Guidance complexity:** approximately `O(BK + JKR)`.

## Method 06 — DFlash6-Boltzmann

Training-free deterministic exploration over the retained top-`K` set. Effective temperature decreases when the top-1/top-2 draft margin is large.

```text
score(c) = z(c) / T_i + deterministic_gumbel(context, position, token_id)
```

**Guidance complexity:** `O(BK)` with no additional neural forward pass.

## Method 07 — DFlash6-BMOBS

Use deterministic Boltzmann scoring at one uncertain middle anchor and fill the remaining positions with MOBS.

**Guidance complexity:** `O(BK)`.

## Method 08 — DFlash7-ACT

ACT uses the already-computed draft margin to shorten an uncertain suffix before LFM2.5 verification. The margin threshold is calibrated on prompts that are separate from the benchmark prompts.

**Guidance complexity:** `O(B)` routing; no additional model forward pass.

## Method 09 — V9 DSpark-Lite

A frozen DFlash backbone is augmented with:

- a low-rank previous-token Markov correction; and
- a scalar prefix-survival confidence head.

The survival confidence decides how much of the proposed suffix is worth verifying.

**Guidance complexity:** approximately `O(BK × rank)`.

## Method 10 — V10 Advanced Boltzmann

Confident slots remain on the ordinary argmax path. Candidate exploration is spent only on a bounded set of uncertain positions. Configuration is chosen by a bounded calibration procedure on separate prompts.

**Guidance complexity:** sparse `O(BK)`; training steps = `0`.

## Method 11 — V11 Boltzmann-Gated MOBS

A deterministic Boltzmann uncertainty score is used as a **routing signal**, not as an authority. Only a bounded number of uncertain slots receive MOBS pair scoring; confident slots remain DFlash argmax.

**Guidance complexity:** approximately `O(MK)` with `M ≤ B` selected slots.

## Method 12 — DFlash12-PARAREAL

V12 introduces **parallel linear residual correction in continuous top-`K` score space**.

### Coarse/fine mapping

```text
G = DFlash top-k block score field
F = frozen LFM2.5 teacher score field on the same candidate IDs
```

`F` is available only during preparation. V12 learns a small ridge-regression approximation to the fine-minus-current residual.

Token IDs are never added or subtracted.

### Closed-form residual model

For standardized feature matrix `X` and residual target `y`:

```text
β = (XᵀX + λI)⁻¹ Xᵀy
```

with:

```text
y = center(F) - q
```

The intercept is not regularized.

### Repeated parallel correction

```text
q₀ = center(G)
Δ_k = clip(R_linear(features(q_k, G)), -c, +c)
q_(k+1) = center(q_k + damping · Δ_k)
```

Every `B × K` candidate row is corrected in vectorized linear algebra. The default uses two correction rounds.

### Features

The eight default features are:

1. current centered candidate score;
2. original coarse centered score;
3. normalized candidate rank;
4. normalized block position;
5. coarse top-1/top-2 margin;
6. candidate similarity to the last prefix-token embedding;
7. candidate similarity to the prefix-mean embedding;
8. current-score × block-position interaction.

### Convergence diagnostics

During preparation and internal holdout evaluation:

```text
E_k = mean((center(F) - q_k)²)
```

The artifact stores `E_k`, `log(E_k)`, contraction ratios `E_(k+1)/E_k`, and fine top-1 agreement. These diagnostics are not inference-time oracles.

**Correction complexity:** `O(RBKD)` plus `O(BKH)` vectorized embedding similarities. V12 adds no target forward pass and no neural correction forward pass.

## Canonical complexity summary

```text
00 Normal:                 O(N) target steps
01 DFlash:                 parallel block draft + verify
02 DFlash2:                O(BK²)
03 DFlash3-MOBS:           O(BK)
04 DFlash4-JUMP:           O(BK + JK) + jump forward
05 DFlash5-FUSED:          O(BK + JKR)
06 DFlash6-Boltzmann:      O(BK)
07 DFlash6-BMOBS:          O(BK)
08 DFlash7-ACT:            O(B) routing
09 V9 DSpark-Lite:         O(BK × rank)
10 V10 Advanced Boltzmann: sparse O(BK)
11 V11 Gated-MOBS:         O(MK), M ≤ B
12 V12 PARAREAL:           O(RBKD) + O(BKH)
```

These expressions describe guidance/correction work, not total Transformer inference cost.

## Exactness and interpretation

A higher acceptance rate, lower teacher-space error, or fewer target calls is **not** sufficient to claim a speedup. The repository reports end-to-end wall-clock throughput under the same LFM2.5 workload and preserves negative results.

The GitHub Pages site is generated from the unified LFM2.5 workflow and reports every method in the same run.

See [`version12-parareal.md`](version12-parareal.md) for the complete V12 derivation and [`reproducibility.md`](reproducibility.md) for the benchmark protocol.
