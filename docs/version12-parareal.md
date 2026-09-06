# DFlash12-PARAREAL: parallel linear residual correction

## Status

DFlash12-PARAREAL is an experimental algorithm introduced in this repository and evaluated only in the canonical **LFM2.5-350M All-12** study.

It is inspired by the coarse/fine residual-correction structure of Parareal for time-parallel numerical integration. It is **not** an implementation of the classical ODE Parareal solver, and no mathematical equivalence is claimed.

The design goal is specific: start from a cheap parallel DFlash block, learn a very small approximation to the target-minus-coarse residual, apply that correction to every retained candidate in parallel, and keep exact target verification.

## 1. Parareal mapping

| Parareal concept | DFlash12 interpretation |
|---|---|
| coarse propagator `G` | DFlash top-k candidate score field |
| fine propagator `F` | frozen LFM2.5 teacher scores, preparation only |
| fine-minus-coarse residual | target score field minus current corrected score field |
| correction iteration | one vectorized affine residual update over all `B × K` rows |
| convergence error | teacher-space top-k score MSE |
| time intervals | speculative block positions |
| final physical solution | exact target-verified continuation |

The correction is performed in **continuous score space**, not token-ID space. Token IDs are categorical identifiers and are never added or subtracted.

## 2. State representation

For block length `B` and retained candidate width `K`, DFlash provides:

```text
G ∈ R^(B × K)
```

Scores are centered per position because an additive logit offset does not change ranking:

```text
center(z_i) = z_i - mean(z_i)
```

The initial state is:

```text
q₀ = center(G)
```

During preparation, the frozen LFM2.5 target supplies `F` on the **same retained candidate IDs**.

## 3. Linear residual surrogate

V12 deliberately uses ordinary standardized ridge regression rather than another neural decoder.

For feature matrix `X` and target residual `y`:

```text
β = (XᵀX + λI)⁻¹ Xᵀy
```

with:

```text
y = center(F) - q
```

The intercept is not regularized. The default ridge coefficient is `0.001`.

This gives V12 a compact correction path that can be evaluated as vectorized linear algebra.

## 4. Feature contract

Each `(block position, retained candidate)` row receives eight features:

1. current centered candidate score;
2. original coarse centered candidate score;
3. normalized rank in the DFlash top-k list;
4. normalized block position;
5. coarse top-1/top-2 margin;
6. candidate-embedding similarity to the last prefix token;
7. candidate-embedding similarity to the prefix-mean embedding;
8. current-score × block-position interaction.

Feature means and scales are learned from the training examples and stored with the coefficients in the V12 JSON artifact.

The embedding similarities are vectorized lookup/dot-product features; they do not require another Transformer forward pass.

## 5. Repeated correction

The inference update is:

```text
Δ_k = clip(R_linear(features(q_k, G)), -c, +c)
q_(k+1) = center(q_k + ω · Δ_k)
```

where the canonical defaults are:

```text
correction rounds R = 2
damping ω           = 0.75
residual clip c     = 6.0
```

All `B × K` rows are evaluated together. There is no dependency on a newly selected token from another block position inside the correction rounds.

After the final correction, V12 selects the highest corrected candidate at every slot and sends the proposal to the ordinary LFM2.5 target verifier.

## 6. Intermediate-state training

A second correction round would be extrapolation if the regressor were trained only at `q₀ = G`. V12 therefore adds interpolated training states:

```text
q_τ = (1 - τ)G + τF
```

with canonical values:

```text
τ = 0.00, 0.50, 0.75
```

Each interpolated state uses target residual:

```text
F - q_τ
```

so repeated correction is trained behavior rather than an accidental repeated application.

## 7. LFM2.5 teacher-data preparation

The canonical V12 artifact is prepared from:

```text
real_benchmarks/train_seeds.json
```

For each frozen LFM2.5 greedy trajectory:

1. generate the deterministic continuation;
2. perform one full causal target pass over the completed trajectory;
3. reuse those causal logits for legal block windows;
4. compute DFlash coarse block logits for each prefix;
5. retain DFlash top-k candidate IDs and scores;
6. index the target logits on the same candidate IDs;
7. compute embedding-similarity features;
8. build coarse/fine regression examples;
9. reserve an internal holdout split;
10. solve the ridge regression in closed form.

The resulting artifact is:

```text
lfm-artifacts/v12_parareal.json
```

It contains no target-model weights.

## 8. Convergence diagnostics

Where teacher `F` is available, V12 records:

```text
E_k = mean((center(F) - q_k)²)
```

and stores:

- `teacher_mse_by_round`;
- `log_teacher_mse_by_round`;
- `contraction_ratio_by_round = E_(k+1) / E_k`;
- fine top-1 agreement by round.

A descending approximately linear trend in `log(E_k)` is the desired signature of geometric contraction. These diagnostics are preparation/holdout measurements and are never inference-time oracles.

## 9. Exactness contract

The V12 linear corrector is never authoritative.

The LFM2.5 target verifier:

1. evaluates the proposed block;
2. accepts only the matching prefix;
3. supplies the first mismatching greedy token;
4. continues until the requested output length is reached.

The unified benchmark then compares the complete output with normal target-only greedy decoding. V12 is valid only when `all_exact == true`.

## 10. Complexity

Let:

- `B` = speculative block length;
- `K` = retained candidates per position;
- `D` = feature count (`8`);
- `R` = correction rounds (`2` by default);
- `H` = target embedding width.

Affine correction work is approximately:

```text
O(R · B · K · D)
```

Embedding similarity work is approximately:

```text
O(B · K · H)
```

V12 adds no target forward pass and no neural correction forward pass. End-to-end speed must still be measured because memory traffic, feature construction, drafter cost, and target verification dominate practical runtime behavior.

## 11. Prepare V12

```bash
python -m dflash_mini_lab.v12_prepare \
  --aux lfm-artifacts/lfm_aux.pt \
  --output lfm-artifacts/v12_parareal.json \
  --seeds real_benchmarks/train_seeds.json \
  --max-seed-count 24 \
  --generation-tokens 24 \
  --top-k 8 \
  --correction-rounds 2 \
  --damping 0.75 \
  --ridge 0.001 \
  --holdout-fraction 0.20 \
  --cpu-threads 2
```

## 12. Evaluate V12 with all other methods

V12 is not published from a separate synthetic/unit benchmark. It is measured in the same real-model run as the other eleven speculative methods:

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

The report page then shows V12 beside DFlash through V11 using the same LFM2.5 target, prompts, CPU protocol, exactness check, and aggregation rule.

## 13. Claims supported by the implementation

The implementation supports these mechanism-level statements:

- a DFlash top-k block can be treated as a coarse continuous score field;
- a fine-minus-current residual can be approximated with ordinary linear regression;
- the residual can be applied to every retained candidate in parallel;
- intermediate coarse/fine states can train repeated correction;
- teacher-space log-error contraction can be measured explicitly;
- exact greedy output is preserved by authoritative target verification when the exactness gate passes.

## 14. Claims requiring measured LFM2.5 evidence

Do not claim any of the following unless the generated all-12 `benchmark.json` supports it:

- V12 is faster than normal greedy decoding;
- V12 is faster than another speculative method;
- V12 improves acceptance;
- V12 has geometric contraction on real holdout data;
- fewer target calls translate to better wall-clock throughput.

The repository intentionally preserves negative results.

## Reference inspiration

V12 is motivated by the coarse/fine correction principle described in the Parareal literature, including **“Parareal Contribution to Speeding-Up the Solving of Nonlinear Ordinary Differential Equations on Parallel/Multi-Core Platforms for Sensing Systems.”** The adaptation here operates on speculative score fields and uses a learned linear surrogate for the fine-minus-current residual.

See [`algorithm.md`](algorithm.md) for the complete 12-method comparison and [`reproducibility.md`](reproducibility.md) for the one-model benchmark protocol.
