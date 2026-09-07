# V13 MinOp and V14 Simple PARAREAL

This note documents the two low-operation experiments added after the canonical LFM2.5 All-12 study.

## Motivation from the measured LFM2.5 run

The preceding benchmark showed two useful facts:

- plain DFlash had the highest measured throughput;
- V10 achieved slightly better acceptance/target-pass efficiency with very little explicit candidate guidance, suggesting that its remaining opportunity was implementation overhead rather than a need for a larger selector;
- V12 reduced teacher-space MSE but did not improve top-1 agreement or acceptance enough to justify its full-field correction cost.

V13 and V14 therefore optimize for **minimum additional inference work**.

---

## V13 MinOp

### Goal

Preserve V10's useful sparse top-2 policy while removing avoidable full-logit NumPy transfer and post-hoc top-k work.

### Runtime path

V10 conceptually does:

```text
Torch drafter -> full B×V logits -> NumPy -> top-2 search -> sparse decision
```

V13 does:

```text
Torch drafter -> torch.topk(k=2) -> transfer B×2 -> one sparse decision
```

The drafter itself is unchanged. V13 changes only how the already-produced logits are reduced and routed.

### Policy isolation

V13 reuses the configuration selected for V10:

- temperature;
- margin cutoff;
- margin slope.

V13 hard-limits the routing budget to one position per speculative block. It is **not separately calibrated**. This makes the benchmark useful for answering a narrow question:

> Can V10-like proposal quality be retained while reducing selector/data-movement overhead?

### Selection rule

1. Obtain top-1 and top-2 candidate IDs/scores for each block position directly in Torch.
2. Compute the top1-top2 margin for the `B` positions.
3. Keep every confident position on draft top-1.
4. Among positions below the V10 margin cutoff, choose only the least-confident position.
5. Apply the same deterministic two-way Boltzmann decision used by V10.
6. Verify the resulting block with the authoritative LFM target.

### Complexity

After top-2 extraction, V13 routing is `O(B)` and the stochastic candidate decision scores only two candidates at at most one slot.

The benchmark records:

- candidate scores;
- routed positions;
- fast-argmax positions;
- number of top-2 values transferred;
- draft and selector time separately.

---

## V14 Simple PARAREAL

### Goal

Keep the coarse/fine residual-correction idea of V12 but reduce the state from a complete `B×K` score field to one scalar per block position.

### Coarse and fine variables

For the same draft top-1 and top-2 candidates:

```text
G = draft_logit(top1) - draft_logit(top2)
F = target_logit(top1) - target_logit(top2)
```

`F` is used only during preparation.

A positive fine gap means the frozen target prefers draft top-1 over draft top-2; a negative fine gap means the target prefers draft top-2 among those two candidates.

### Fine estimator

V14 fits a three-term affine estimator:

```text
F_hat = a + b·G + c·p
```

where `p` is normalized block position.

The coefficients are fit with closed-form ridge regression on frozen LFM teacher trajectories. No neural estimator is trained.

### One-round Parareal update

At inference:

```text
gap_0 = G
gap_1 = G + damping · (F_hat - G)
```

This is a scalar coarse/fine residual update. V14 then allows at most one position per block to switch from draft top-1 to draft top-2, choosing the most negative corrected gap when it crosses the switch threshold.

### Why only top-2?

The goal of V14 is not to approximate the complete target distribution. It asks a narrower question relevant to low-operation speculative decoding:

> Can a tiny estimate of the target's local top-2 preference identify the single DFlash slot most worth correcting?

This drastically reduces correction work compared with V12.

### Artifact

The default artifact is:

```text
lfm-artifacts/v14_simple_parareal.json
```

It contains:

- three estimator coefficients;
- damping, switch threshold and correction budget;
- model/block metadata;
- train and holdout fine-gap diagnostics.

Target weights are not redistributed.

### Diagnostics

V14 preparation records:

- fine-gap MSE;
- fine-gap MAE;
- corrected sign accuracy;
- coarse sign accuracy;
- rate at which the teacher prefers top-2;
- predicted switch rate;
- switch precision and recall.

These are estimator diagnostics, not evidence of end-to-end speedup. Throughput and exactness must come from the canonical LFM2.5 benchmark.

---

## Exactness contract

Neither V13 nor V14 is authoritative. As with every speculative method in the lab, the LFM target verifier accepts only the matching prefix and supplies the target token at the first mismatch. The complete output must exactly match normal greedy LFM2.5.

## Canonical comparison

The All-14 workflow measures V13 and V14 together with the earlier methods using the same prompts, target, verifier, CPU settings, output length and rotated execution order.

The most important direct comparisons are:

```text
V10 vs V13  -> same policy family, less runtime plumbing
V12 vs V14  -> full-field residual vs scalar fine-gap residual
```

Do not claim V13 or V14 is faster until the corresponding `benchmark.json` from the real LFM2.5 run supports that claim.
