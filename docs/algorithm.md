# Algorithm notes

This repository is a **mechanism-level CPU reference lab**, not the upstream DFlash inference engine. It now compares five decoding paths under one deterministic workload.

## 1. Normal autoregressive decoding

For each output token the target SLM runs on the visible sequence, chooses the greedy next token and appends it. For `N` new tokens the reference path performs `N` target forward passes.

## 2. DFlash reference mode

A small non-causal drafter receives cheap context features derived from the target embedding table and predicts all positions of a future block in one draft forward pass. The target verifies the proposal in one pass, accepts the matching prefix and corrects the first mismatch.

The upstream DFlash design uses a lightweight block-diffusion drafter conditioned on target hidden features. This lab preserves the parallel-draft → exact-target-verify mechanism at a tiny CPU scale.

## 3. DFlash v2 reference mode

The DFlash2-style selector keeps top-k candidates at every draft position and uses learned low-rank predecessor-conditioned transition scores with dynamic programming.

For block length `B` and top-k width `K`, the measured transition work is approximately:

```text
K + (B - 1) * K^2
```

so selector work grows as **O(BK²)**.

## 4. Experimental DFlash3-MOBS

**MOBS = Middle-Out Bidirectional Selection.** It avoids constructing the full `K × K` transition grid.

1. choose a reproducible pseudo-random central anchor;
2. expand left and right;
3. at each new position score only `K` candidates against already selected neighbor(s);
4. optionally run a fixed odd/even local refinement pass;
5. send the path to the exact target verifier.

With a constant refinement count, core selector work is **O(BK)**. The CPU fast-path benchmark disables refinement because earlier ablations showed that its extra overhead did not consistently pay for itself.

## 5. Experimental DFlash4-JUMP-MOBS

JUMP-MOBS tests a different way to recover global path quality without returning to the DFlash2 `K × K` lattice.

### A. Sparse indexed future-token head

A separately trained tiny MLP receives the same context features plus a learned offset embedding and predicts a token distribution directly at sparse future indexes.

For the bundled block size `B=4`, the jump offsets are:

```text
+2, +4
```

Conceptually the jump head models:

```text
P(x[t+j] | h_t, j)
```

for sparse `j`, rather than generating all intermediate tokens autoregressively.

### B. Anchor selection

At each sparse jump position, JUMP-MOBS restricts the choice to the parallel drafter's top-k candidates and combines the draft score with the jump-head score:

```text
anchor_score(candidate)
    = draft_score(candidate)
    + jump_weight * jump_score(candidate)
```

The selected jump tokens are approximate anchors. They are **not trusted output tokens**.

### C. O(BK) gap filling

After the sparse anchors are chosen, remaining positions are filled in wavefronts. Each unanchored position is scored once against at most two already selected adjacent neighbors.

For the four-token reference block:

```text
position:   +1      +2      +3      +4
                     ^                ^
                  jump anchor      jump anchor
                    /  \              /
                 fill +1          fill +3
```

The implementation counts both:

- local learned pair scores;
- jump-anchor candidate scores.

For a fixed sparse jump set `J`, guidance work is approximately:

```text
O(BK + JK)
```

and because `J << B` in the intended design, this remains effectively **O(BK)** rather than O(BK²).

### D. Exact target verification

The jump head can be wrong. The selected anchors can be wrong. The complete JUMP-MOBS proposal is still sent through the same target verifier. The final greedy output is therefore checked against normal target-only decoding exactly like the other speculative modes.

## What the benchmark is actually testing

Lower selector complexity is useful only if proposal quality remains high enough to reduce expensive target verification calls. JUMP-MOBS additionally pays the latency of one tiny jump-head forward pass per speculative block.

The benchmark therefore reports:

- median tokens/sec and latency;
- draft acceptance;
- tokens per target pass;
- target/draft/jump forward-pass counts;
- local selector pair scores;
- jump candidate scores;
- total guidance scores;
- exact-output equivalence.

The useful objective is not merely the lowest asymptotic selector cost. It is:

```text
verified output tokens / total decode time
```

A JUMP-MOBS run may therefore improve acceptance and reduce guidance work yet still lose wall-clock throughput if the jump head costs more than the saved target/selector work.

## Complexity summary

```text
Normal:             O(N) target passes for N output tokens
DFlash:             parallel draft + target verification
DFlash2 selector:   O(BK^2)
DFlash3-MOBS:       O(BK)
DFlash4-JUMP-MOBS:  O(BK + JK)  ≈ O(BK) for sparse J
```

These complexity statements refer to path-guidance candidate scoring in this reference implementation, not total Transformer inference complexity.

## Exactness guarantee in this lab

Tests and CI compare the complete greedy token sequence from all speculative modes against the normal target-only greedy sequence. The speculative proposal is never emitted without target verification/correction.
