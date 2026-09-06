# Algorithm notes

This repository is a mechanism-level CPU research/reference lab. It contains several speculative-decoding experiments under a shared exact-verification contract. Experimental versions introduced here are not upstream DFlash/DFlash2/EAGLE algorithms.

## 1. Normal autoregressive decoding

The target model runs on the visible sequence, chooses the greedy next token, appends it, and repeats. This path defines the exact reference output.

## 2. DFlash reference mode

A small non-causal drafter predicts every position of a future block in one forward pass. The target verifies the block, accepts the matching prefix, and corrects the first mismatch.

## 3. DFlash2-style reference mode

The selector retains top-k candidates at every position and uses predecessor-conditioned transition scores plus dynamic programming. Guidance work is approximately:

```text
K + (B - 1) * K^2
```

or `O(BK^2)`.

## 4. DFlash3-MOBS

MOBS chooses a central anchor and expands left/right, scoring only `K` candidates against already selected neighbors. With a fixed number of refinement passes its selector work is `O(BK)`.

## 5. DFlash4-JUMP-MOBS

A separately trained jump head predicts sparse future anchors, then `O(BK)` local gap filling constructs the complete path. The measured CPU weakness is the extra jump-head forward pass.

## 6. DFlash5-FUSED-JUMP-MOBS

DFlash5 reuses existing drafter hidden states and applies a low-rank residual only to retained top-k candidates at sparse offsets. It removes DFlash4's separate jump-forward pass while preserving exact target verification.

## 7. DFlash6-Boltzmann

DFlash6-Boltzmann is training-free. It uses only the draft logits already computed by DFlash.

At each position, conceptually:

```text
P(c) proportional to exp(z(c) / T_i)
```

The implementation uses deterministic Gumbel-Max:

```text
score(c) = z(c) / T_i + gumbel(context, position, token_id)
```

The Gumbel value is generated deterministically from context/token IDs. Effective temperature falls as the top-1/top-2 draft margin increases. Candidate scoring is `O(BK)` and adds no model forward pass.

## 8. DFlash6-BMOBS

BMOBS combines one Boltzmann-selected middle anchor with the previously tested middle-out linear selector. Remaining positions are filled with adjacent-neighbor MOBS scoring. Guidance remains approximately `O(BK)`.

## 9. DFlash7-ACT

ACT is the cached-Qwen adaptive speculation experiment. It uses the already-computed draft top-1/top-2 margin to shorten an uncertain speculative suffix before the expensive target verifier. It adds no neural forward pass and can fall back to fixed DFlash when its threshold is zero.

## 10. V8, V9, V10 and V11 real-model experiments

The later LFM/EAGLE-oriented experiments are kept separate from the original tiny-path numbering:

- **V8**: EAGLE3-oriented comparison path;
- **V9 DSpark-Lite**: frozen DFlash backbone plus low-rank previous-token Markov correction and prefix-survival confidence;
- **V10 Advanced Boltzmann**: cost-aware training-free candidate exploration;
- **V11 Boltzmann-Gated MOBS**: uses deterministic Boltzmann uncertainty only as a routing signal, leaving confident slots at DFlash argmax and sending a bounded number of uncertain slots through MOBS.

V11 reduces path-guidance work by applying pair scoring sparsely rather than at every position.

## 11. DFlash12-PARAREAL

DFlash12 introduces a new correction family: **parallel linear residual refinement in continuous top-k score space**.

### 11.1 Coarse/fine interpretation

The Parareal-inspired mapping is:

```text
G = DFlash top-k block logits
F = frozen target teacher logits on the same candidate IDs (preparation only)
```

The model learns a compact approximation to `F - q`, where `q` is the current corrected score field.

Token IDs are never added or subtracted. The correction state is a centered real-valued logit field of shape `B x K`.

### 11.2 Linear regression

For each candidate row, V12 builds eight standardized features and solves ridge regression in closed form:

```text
beta = (X^T X + lambda I)^(-1) X^T y
```

with target:

```text
y = center(F) - q
```

The intercept is left unregularized.

### 11.3 Intermediate-state training

To make repeated correction a trained behavior, each teacher block contributes interpolated states:

```text
q_tau = (1 - tau) * G + tau * F
```

Default `tau` values are `0.00, 0.50, 0.75`.

### 11.4 Parallel correction

At inference:

```text
q_0 = center(G)
Delta_k = clip(R_linear(features(q_k, G)), -c, +c)
q_(k+1) = center(q_k + damping * Delta_k)
```

All `B x K` rows are evaluated with vectorized linear algebra. There is no newly selected token dependency between positions inside the correction loop.

### 11.5 Features

The default features are:

1. current centered score;
2. original coarse centered score;
3. normalized candidate rank;
4. normalized block position;
5. original top-1/top-2 coarse margin;
6. candidate-to-last-token embedding similarity;
7. candidate-to-prefix-mean embedding similarity;
8. current-score × position interaction.

The semantic similarities use embedding lookup and vectorized dot products, not a model forward pass.

### 11.6 Convergence diagnostics

Preparation records:

```text
E_k = mean((center(F) - q_k)^2)
```

plus `log(E_k)`, contraction ratios `E_(k+1)/E_k`, and fine top-1 agreement by round. A descending near-linear trend in `log(E_k)` is the desired geometric-convergence signature.

These diagnostics are available only where teacher `F` exists. They are not used as an inference-time oracle.

### 11.7 Complexity

With `D=8` linear features and `R` correction rounds, the affine correction is approximately:

```text
O(R * B * K * D)
```

The two embedding-similarity features add approximately:

```text
O(B * K * H)
```

for hidden width `H`. V12 adds no target forward pass and no neural correction forward pass.

## Exactness contract

Every speculative method in this repository is approximate until verified. Final greedy exactness comes from the target verifier, which accepts only the matching prefix and inserts the target's first mismatching token. Benchmarks compare complete outputs with normal target-only greedy decoding.

A candidate-selection improvement, acceptance increase, or lower teacher-space error is not sufficient to claim end-to-end speedup.

## Complexity summary

```text
Normal:                    O(N) target passes for N output tokens
DFlash:                    parallel draft + target verification
DFlash2 selector:          O(BK^2)
DFlash3-MOBS:              O(BK)
DFlash4-JUMP-MOBS:         O(BK + JK) plus separate jump inference
DFlash5-FUSED-JUMP:        O(BK + JKR), no separate jump inference
DFlash6-Boltzmann:         O(BK), no learned selector/model pass
DFlash6-BMOBS:             O(BK), one Boltzmann anchor + linear fill
V9 DSpark-Lite:            O(BK * rank) low-rank Markov correction
V11 gated MOBS:            O(MK), M <= B selected uncertain slots
V12 affine residual:       O(RBKD) + O(BKH) similarity features
```

These expressions describe guidance/correction work in this reference implementation, not total Transformer inference complexity.

## Interpretation discipline

The repository preserves negative results. A method is not described as faster simply because it uses fewer target calls or raises draft acceptance. Only measured end-to-end wall-clock results under the same workload support a speed claim.

See [`version12-parareal.md`](version12-parareal.md) for the complete V12 design and research protocol.
