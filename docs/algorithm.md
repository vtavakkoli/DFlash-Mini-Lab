# Algorithm notes

This repository is a **mechanism-level CPU reference lab**, not the upstream DFlash inference engine.
It is designed to make the three decoding paths inspectable and benchmarkable under one deterministic workload.

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
3. selects a coherent candidate path across the block;
4. sends that path to the same lossless target verifier.

This mirrors the central public DFlash2 idea: a small selector adds predecessor-conditioned candidate scoring after the parallel draft pass. The current upstream DFlash2 design also includes local dynamic convolutions and other inference/training details that are **not reproduced here**.

## Why the DFlash v2 CPU throughput can be lower than DFlash

The selector can increase accepted tokens per verification pass while still reducing raw wall-clock throughput for a tiny CPU model. In this repository the selector runs in NumPy/Python, and its overhead is large relative to the deliberately tiny target model. That is a useful result: **higher acceptance does not automatically imply higher end-to-end speed**.

On production accelerators and large target models, the cost balance is very different.

## Exactness guarantee in this lab

The tests compare the complete greedy token sequence from DFlash and DFlash v2 against the normal target-only greedy sequence. The speculative proposal is never trusted without target verification.
