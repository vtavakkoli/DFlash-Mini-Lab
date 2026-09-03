# Algorithm notes

This repository is a **mechanism-level CPU reference lab**, not the upstream DFlash inference engine. It compares six decoding paths under one deterministic workload.

## 1. Normal autoregressive decoding

For each output token the target SLM runs on the visible sequence, chooses the greedy next token and appends it. For `N` new tokens the reference path performs `N` target forward passes.

## 2. DFlash reference mode

A small non-causal drafter receives cheap context features derived from the target embedding table and predicts all positions of a future block in one draft forward pass. The target verifies the proposal in one pass, accepts the matching prefix and corrects the first mismatch.

The upstream DFlash design uses a lightweight block-diffusion drafter conditioned on target hidden features. This lab preserves the parallel-draft → exact-target-verify mechanism at a tiny CPU scale.

## 3. DFlash v2 reference mode

The DFlash2-style selector keeps top-k candidates at every draft position and uses learned low-rank predecessor-conditioned transition scores with dynamic programming.

For block length `B` and top-k width `K`, measured transition work is approximately:

```text
K + (B - 1) * K^2
```

so selector work grows as **O(BK²)**.

## 4. Experimental DFlash3-MOBS

**MOBS = Middle-Out Bidirectional Selection.** It avoids constructing the full `K × K` transition grid.

1. choose a reproducible pseudo-random central anchor;
2. expand left and right;
3. at each new position score only `K` candidates against selected neighbor(s);
4. optionally run a fixed odd/even local refinement pass;
5. send the path to the exact target verifier.

With a constant refinement count, core selector work is **O(BK)**. The CPU fast path disables refinement because earlier ablations showed its overhead did not consistently pay for itself.

## 5. Experimental DFlash4-JUMP-MOBS

DFlash4 adds a separately trained indexed future-token head. For block size `B=4`, the jump offsets are `+2,+4`.

Conceptually it models:

```text
P(x[t+j] | h_t, j)
```

At each jump position, the choice is restricted to the drafter's top-k candidates and combines the draft score with the jump-head score. Those approximate anchors then seed O(BK) local gap filling.

For a fixed sparse jump set `J`, guidance work is approximately:

```text
O(BK + JK)
```

The weakness found by the CPU benchmark is that DFlash4 pays **one separate jump-head forward pass per speculative block**. The extra inference can erase the savings from better proposals.

## 6. Experimental DFlash5-FUSED-JUMP-MOBS

DFlash5 targets that measured DFlash4 bottleneck.

### A. Reuse the existing drafter computation

The normal parallel drafter already computes a hidden vector for every future block slot before projecting to vocabulary logits. DFlash5 exposes those normalized drafter hidden states:

```text
context
   │
   ▼
parallel drafter ──► hidden(+1,+2,+3,+4)
   │                       │
   │                       └──► fused jump residual at +2/+4
   ▼
normal draft logits
```

There is **no second drafter/MLP/Transformer forward pass** for DFlash5.

### B. Candidate-only low-rank residual

At sparse positions `+2,+4`, a low-rank query is computed from the already-available drafter hidden state plus an offset embedding. It is scored only against the candidate codebook rows for the `K` tokens retained by the normal drafter.

For sparse anchor count `J` and residual rank `R`, this extra scoring is:

```text
O(J*K*R)
```

rather than a full-vocabulary jump projection.

### C. Teacher distillation is training-only

The deterministic builder can train the fused residual using the stronger DFlash4 jump distribution as a teacher. The teacher contributes only during model building. At inference:

```text
DFlash4 jump forward passes: > 0
DFlash5 jump forward passes:   0
```

The benchmark records that difference explicitly.

### D. Confidence-gated anchors

DFlash5 combines the normal draft score with the fused residual at each sparse position. A configurable top-1/top-2 margin can reject weak anchors. If no fused anchor is retained, the method falls back to the MOBS path.

A bounded CI sweep tests a fixed set of residual weights and margins and chooses the fastest configuration that remains exact and keeps guidance work below DFlash2.

### E. O(BK) gap filling and exact verification

Once fused anchors are selected, missing positions are filled using the same adjacent-neighbor O(BK) mechanism as JUMP-MOBS. The complete proposal still goes through the target verifier. No fused/jump prediction is emitted directly.

## What the benchmark tests

The useful objective is not merely low selector complexity. It is:

```text
verified output tokens / total decode time
```

The report therefore includes:

- median tokens/sec and latency;
- draft acceptance;
- tokens per target pass;
- target/draft/jump forward-pass counts;
- local selector pair scores;
- jump/fused candidate scores;
- fused-anchor count;
- total guidance scores;
- exact-output equivalence.

This exposes three different failure modes:

1. low guidance cost but weak speculative paths;
2. strong paths but expensive extra inference, as seen with DFlash4;
3. cheap fused guidance whose proposal-quality gain is too small to beat simpler methods.

The project preserves those negative results rather than tuning until a benchmark happens to rank a new method first.

## Complexity summary

```text
Normal:                 O(N) target passes for N output tokens
DFlash:                 parallel draft + target verification
DFlash2 selector:       O(BK^2)
DFlash3-MOBS:           O(BK)
DFlash4-JUMP-MOBS:      O(BK + JK) plus a separate jump-head forward
DFlash5-FUSED-JUMP:     O(BK + JKR), no separate jump-head forward
```

These statements refer to path-guidance candidate scoring in this reference implementation, not total Transformer inference complexity.

## Exactness guarantee

Tests and CI compare the complete greedy token sequence from all speculative modes against the normal target-only greedy sequence. The speculative proposal is never emitted without target verification/correction.
