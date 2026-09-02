# Algorithm notes

This repository is a **mechanism-level CPU reference lab**, not the upstream DFlash inference engine.
It is designed to make four decoding paths inspectable and benchmarkable under one deterministic workload.

## 1. Normal autoregressive decoding

For each output token:

1. run the target SLM on the visible sequence;
2. select the greedy next token;
3. append it;
4. repeat.

For `N` new tokens, the reference path therefore performs `N` target forward passes.

## 2. DFlash reference mode

The bundled small drafter receives a cheap target-context feature derived from the target embedding table.
A non-causal Transformer processes all block slots together and emits logits for the whole future block in **one draft forward pass**.

The target then verifies the proposed block in one target pass. The decoder accepts the longest matching prefix and uses the target's token at the first disagreement. This greedy verifier correction makes the final output identical to normal greedy decoding.

The upstream DFlash design uses a lightweight block-diffusion drafter conditioned on target hidden features and predicts the complete block in one parallel forward pass.

## 3. DFlash v2 reference mode

DFlash v2 keeps the same parallel draft pass but does not immediately take one independent argmax per slot.
Instead, this lab:

1. retains the top-k candidates at every draft position;
2. computes a learned low-rank predecessor-conditioned transition score;
3. selects a coherent candidate path across the block with dynamic programming;
4. sends that path to the same lossless target verifier.

For block length `B` and top-k width `K`, the transition grid evaluates approximately:

```text
K + (B - 1) * K^2
```

pair scores, so selector work grows as **O(BK²)**.

This mirrors the central public DFlash2 idea: a small selector adds predecessor-conditioned candidate scoring after the parallel draft pass. The current upstream DFlash2 design also includes local dynamic convolutions and other inference/training details that are **not reproduced here**.

## 4. Experimental DFlash3-MOBS

**MOBS = Middle-Out Bidirectional Selection.** This is an experimental algorithm introduced in this repository to test whether a cheaper path selector can trade a small amount of path quality for lower CPU overhead.

The selector reuses the same learned low-rank transition embeddings as the DFlash2 reference mode, but avoids constructing a full `K × K` transition grid between adjacent positions.

### Step A — reproducible random-middle anchor

For an odd block, MOBS starts at the center. For an even block, it uses a cheap deterministic fingerprint of the context and previous token to pseudo-randomly choose one of the two central positions. This preserves benchmark reproducibility while testing the random-middle idea.

### Step B — middle-out bidirectional expansion

After choosing the best anchor candidate, MOBS expands left and right. At each new position it scores only its `K` candidates against the already selected neighboring token.

```text
p1   p2   p3   p4   p5   p6
          ^
        anchor
       /      \
     p2        p4
     /          \
   p1            p5 -> p6
```

This changes the core path-selection work from a `K × K` transition matrix to `K` neighbor comparisons per position.

### Step C — bubble-like odd/even refinement

A fixed number of local refinement passes then re-scores alternating positions against their selected left/right neighbors:

```text
pass A: (p1)   (p3)   (p5)
pass B:    (p2)   (p4)   (p6)
```

This is inspired by the local-update pattern of odd/even bubble-style refinement, but it is **not literal bubble sort**. No global sorting operation is performed. Each candidate is compared only with already selected neighboring tokens.

With a constant number of refinement passes, the selector remains **O(BK)**.

### Selector-work comparison

Ignoring small boundary constants:

```text
DFlash2:      O(BK²)
DFlash3-MOBS: O(BK)
```

For the benchmark the implementation counts the actual learned pair scores evaluated, so the complexity claim is directly testable rather than inferred only from code structure.

## Why lower selector complexity may or may not win

Lower selector work does not guarantee higher end-to-end throughput. A cheaper selector can choose a worse speculative path, lowering draft acceptance and causing more target verification passes. Conversely, a slightly worse path may still win if selector overhead falls enough.

The benchmark therefore reports both:

- median tokens/sec and latency;
- draft acceptance and tokens/target-pass;
- actual selector pair-score count.

The experiment asks whether MOBS improves the useful objective:

```text
verified output tokens / total decode time
```

rather than merely minimizing selector operations.

## Exactness guarantee in this lab

Tests compare the complete greedy token sequence from DFlash, DFlash v2 and DFlash3-MOBS against the normal target-only greedy sequence. All speculative modes use the same target verification/correction rule, so the speculative proposal is never trusted without target verification.
