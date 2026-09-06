# DFlash12-PARAREAL: parallel linear residual correction

## Status

DFlash12-PARAREAL is an experimental algorithm introduced in this repository. It is inspired by the coarse/fine residual-correction structure of the Parareal method for time-parallel numerical integration. It is **not** an implementation of the classical ODE Parareal solver, and no mathematical equivalence is claimed.

The goal is narrower: use a cheap coarse speculative block, learn a compact approximation to the target-minus-coarse residual, apply that correction to all future positions in parallel, and retain exact target verification.

## 1. Mapping from Parareal to speculative decoding

Classical Parareal combines a cheap coarse propagator `G` and an expensive fine propagator `F`. DFlash12 uses the following analogy:

| Parareal concept | DFlash12 interpretation |
|---|---|
| coarse propagator `G` | DFlash top-k candidate logits for the full speculative block |
| fine propagator `F` | frozen target-model logits on teacher trajectories, used only during preparation |
| fine-minus-coarse residual | target top-k score field minus DFlash top-k score field |
| correction iteration | one vectorized affine residual update over all retained candidates |
| convergence error | teacher-space top-k score MSE on preparation/holdout examples |
| time intervals | speculative block positions |
| final physical solution | exact target-verified greedy continuation |

The important design choice is that correction happens in **continuous score space**, not token-ID space. Token IDs are categorical identifiers and do not support meaningful subtraction.

## 2. State and candidate field

Let block length be `B` and retained candidate width be `K`. DFlash produces candidate scores

```text
G in R^(B x K)
```

for a fixed top-k candidate set at each future position. Scores are centered per position because additive logit offsets do not change ranking:

```text
center(z_i) = z_i - mean(z_i)
```

The initial V12 state is

```text
q_0 = center(G)
```

The frozen target supplies the teacher/fine score field `F` during preparation. `F` is evaluated on the **same retained DFlash candidate IDs**, so regression learns how the target would re-score DFlash's own candidate set.

## 3. Linear residual model

V12 fits a standardized ridge regression model

```text
R_beta(x) ~= center(F) - q
```

with a closed-form least-squares solution:

```text
beta = (X^T X + lambda I)^(-1) X^T y
```

The intercept is not regularized. The default ridge coefficient is `1e-3`.

This is intentionally small. V12 does not add a Transformer, MLP, recurrent network, or learned decoding head at inference time.

## 4. Feature vector

For every `(position, candidate)` pair the default feature vector contains eight values:

1. current centered candidate score;
2. original coarse centered candidate score;
3. normalized rank in the original DFlash top-k list;
4. normalized block position;
5. original top-1/top-2 DFlash margin at that position;
6. scaled dot-product similarity between the candidate embedding and the last prefix-token embedding;
7. scaled dot-product similarity between the candidate embedding and the prefix-mean embedding;
8. current score multiplied by normalized block position.

All feature rows are standardized using training-set mean and standard deviation. Those statistics are stored in the JSON artifact.

The two embedding-similarity features are computed with vectorized indexing/dot products. They provide candidate-specific semantic information without another model forward pass.

## 5. Iterative correction

For correction round `k`:

```text
Delta_k = clip(R_beta(features(q_k, G)), -c, +c)
q_(k+1) = center(q_k + omega * Delta_k)
```

where:

- `omega` is the damping factor, default `0.75`;
- `c` is the residual clip, default `6.0`;
- default correction rounds = `2`.

Every `(B x K)` feature row is evaluated in one vectorized NumPy operation. There is no dependency on a newly chosen token from another position, so the correction itself remains parallel across the block.

After the final round, V12 chooses the highest corrected score at every block position and passes the complete proposal to the ordinary target verifier.

## 6. Why training includes intermediate states

If regression were trained only on `q_0 = G`, a second correction round would apply the model outside the state distribution seen during fitting. V12 therefore augments each training block with interpolated states:

```text
q_tau = (1 - tau) * G + tau * F
```

using default interpolation values:

```text
0.00, 0.50, 0.75
```

For each state the regression target is:

```text
F - q_tau
```

This makes repeated residual application a trained behavior rather than an accidental extrapolation.

## 7. Teacher-data collection

Preparation uses frozen target-model greedy trajectories.

For each training seed:

1. generate a deterministic greedy continuation;
2. run one full causal target forward on the completed trajectory;
3. reuse the resulting target logits for every legal block window on that trajectory;
4. compute the DFlash block logits for each prefix;
5. retain DFlash top-k candidate IDs and scores;
6. index the cached teacher logits at the same candidate IDs;
7. compute the two embedding-similarity features;
8. construct coarse/fine regression examples.

Because causal logits at a position do not depend on later tokens, one completed-trajectory target forward can provide the fine logits for all block windows on that trajectory.

## 8. Convergence diagnostics

V12 records teacher-space score error by round:

```text
E_k = mean((center(F) - q_k)^2)
```

and stores:

- `teacher_mse_by_round`;
- `log_teacher_mse_by_round`;
- `contraction_ratio_by_round = E_(k+1) / E_k`;
- fine top-1 agreement by round.

A straight descending trend in `log(E_k)` is the desired signature of approximately geometric convergence. These values are measured only when `F` is available during preparation/evaluation.

They are **not** used during normal inference.

## 9. Exactness contract

V12 proposals are approximate. The linear corrector is never authoritative.

The target verifier:

1. evaluates the proposed block;
2. accepts only the matching prefix;
3. inserts the target's first mismatching greedy token;
4. continues until the requested output length is reached.

Every benchmark row includes an exact comparison with normal target-only greedy decoding. A V12 configuration is not considered valid if this comparison fails.

## 10. Complexity

Let:

- `B` = speculative block length;
- `K` = retained candidates per position;
- `D` = linear feature count (`8` by default);
- `R` = correction rounds (`2` by default);
- `H` = target embedding width.

The affine residual work is approximately:

```text
O(R * B * K * D)
```

The two semantic similarity features add vectorized embedding dot products of approximately:

```text
O(B * K * H)
```

No additional target or neural drafter forward pass is introduced by the V12 correction itself.

This complexity statement concerns only V12 guidance/correction. End-to-end latency is still dominated by target/drafter inference and memory behavior.

## 11. Artifact format

The default artifact is a small JSON file:

```text
lfm-artifacts/v12_parareal.json
```

It stores:

- format version and algorithm name;
- feature contract;
- regression coefficients;
- feature normalization statistics;
- model ID and block/candidate metadata;
- training configuration;
- training and holdout convergence diagnostics.

Target weights are not embedded or redistributed in this artifact.

## 12. Commands

Prepare the regression model:

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

Benchmark it:

```bash
python -m dflash_mini_lab.v12_benchmark \
  --aux lfm-artifacts/lfm_aux.pt \
  --dspark lfm-artifacts/lfm_dspark.pt \
  --v12-model lfm-artifacts/v12_parareal.json \
  --prompts real_benchmarks/test_prompts.json \
  --output-dir v12-reports
```

Run unit tests:

```bash
pytest -q
```

## 13. Research claims that are currently justified

The implementation supports these mechanism-level statements:

- a DFlash top-k block can be treated as a coarse continuous score field;
- a fine-minus-coarse residual can be learned with ordinary linear regression;
- the residual can be applied to all retained candidates in parallel;
- repeated correction can be trained using intermediate coarse/fine states;
- teacher-space log-error convergence can be measured explicitly;
- exact greedy decoding is preserved by target verification.

## 14. Claims that require real benchmark evidence

Do **not** claim any of the following until the corresponding measured V12 report supports it:

- V12 is faster than normal greedy decoding;
- V12 is faster than DFlash or V11;
- V12 improves acceptance on arbitrary prompts;
- V12 converges geometrically on real-model holdout data;
- V12 generalizes across models or candidate vocabularies;
- V12 is equivalent to classical Parareal.

The synthetic convergence unit test validates the numerical mechanism only.

## 15. Recommended next experiments

1. Run the default LFM holdout preparation and inspect `log_teacher_mse_by_round`.
2. Compare one, two, and three correction rounds.
3. Sweep damping over a bounded set such as `0.50, 0.75, 1.00`.
4. Measure correction time separately from drafter and verifier time.
5. Compare V12 against plain DFlash and V11 on the same prompt order and exactness reference.
6. If the linear residual consistently contracts teacher error, port the artifact/selector interface to the cached Qwen path.

## Reference inspiration

V12 is motivated by the coarse/fine correction principle described in the Parareal literature, including the work titled **“Parareal Contribution to Speeding-Up the Solving of Nonlinear Ordinary Differential Equations on Parallel/Multi-Core Platforms for Sensing Systems.”** The adaptation here operates on speculative logit fields and uses a learned linear surrogate for the fine-minus-coarse residual.
