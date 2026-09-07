# LFM2.5 All-14 algorithm notes

The canonical benchmark evaluates **14 speculative-decoding mechanisms against one target model: `LiquidAI/LFM2.5-350M-Base`**. Normal target-only greedy decoding is method `00` and defines both the exact reference output and the `1.000×` throughput baseline.

All speculative proposals remain approximate until the same LFM2.5 target verifier accepts the matching prefix. Complete generated sequences are compared with normal greedy decoding.

> DFlash3 through DFlash14 are experimental lab variants. Their numbers describe this repository's research sequence, not upstream official DFlash releases.

## Method map

| # | Method | Main idea | Guidance/correction complexity |
|---:|---|---|---|
| 00 | Normal | target-only greedy decoding | O(N) target steps |
| 01 | DFlash | parallel future-block argmax | draft + verify |
| 02 | DFlash2 | predecessor-aware top-k DP | O(BK²) |
| 03 | DFlash3-MOBS | middle-out local path construction | O(BK) |
| 04 | DFlash4-JUMP-MOBS | sparse jump anchors + gap fill | O(BK + JK) + jump pass |
| 05 | DFlash5-FUSED-JUMP | reuse drafter hidden state for sparse anchors | O(BK + JKR) |
| 06 | DFlash6-Boltzmann | deterministic top-k exploration | O(BK) |
| 07 | DFlash6-BMOBS | Boltzmann anchor + MOBS fill | O(BK) |
| 08 | DFlash7-ACT | margin-based verifier horizon | O(B) routing |
| 09 | V9 DSpark-Lite | low-rank Markov correction + survival head | O(BK × rank) |
| 10 | V10 Advanced Boltzmann | sparse top-2 uncertainty routing | sparse O(BK) |
| 11 | V11 Gated-MOBS | route only uncertain slots to MOBS | O(MK), M ≤ B |
| 12 | V12 PARAREAL | full-field learned fine-minus-coarse correction | O(RBKD) + O(BKH) |
| 13 | **V13 MinOp** | **fused Torch top-2 + one-slot V10 policy** | **O(B) routing after top-2** |
| 14 | **V14 Simple PARAREAL** | **scalar fine-gap estimator + one residual update** | **O(B)** |

These expressions describe selector/correction work, not total Transformer inference complexity.

## V10 -> V13: optimize the low-operation path

V10 obtains full draft logits in NumPy, performs a top-2 search, then routes only a small number of uncertain positions.

V13 keeps the same policy family but changes the execution path:

```text
V10:
Torch drafter -> B×V NumPy logits -> NumPy top-2 -> sparse decision

V13:
Torch drafter -> torch.topk(2) -> B×2 NumPy values -> one sparse decision
```

The selected V10 temperature, margin cutoff and margin slope are reused. V13 hard-limits the routing budget to one position, so the V10→V13 comparison measures whether less data movement and simpler routing can preserve the useful proposal behavior.

For the routed slot, V13 uses a deterministic two-way Boltzmann probability based on the top1-top2 margin. All non-routed slots remain ordinary DFlash top-1.

## V12 -> V14: compress the Parareal state

V12 operates on a continuous `B×K` score field. Its coarse/fine mapping is:

```text
G = DFlash top-k block score field
F = frozen LFM target score field on the same candidates, preparation only
```

It learns a multi-feature residual and applies repeated correction rounds.

V14 keeps the coarse/fine residual idea but reduces each block position to one scalar:

```text
G = draft_logit(top1) - draft_logit(top2)
F = target_logit(top1) - target_logit(top2)   # preparation only
```

The fine estimator is:

```text
F_hat = a + b·G + c·p
```

where `p` is normalized block position. The three coefficients are fit by closed-form ridge regression.

The one-round Parareal-style update is:

```text
gap_0 = G
gap_1 = G + damping · (F_hat - G)
```

If the corrected gap crosses the switch threshold, at most the single most negative slot may change from draft top-1 to draft top-2.

This changes the research question from:

> Can we reconstruct the full fine score field?

to:

> Can we cheaply identify the one DFlash top-2 correction most worth trying?

## V14 estimator diagnostics

The V14 preparation artifact records:

- fine-gap MSE and MAE;
- corrected sign accuracy;
- coarse sign accuracy;
- teacher top-2 preference rate;
- predicted switch rate;
- switch precision and recall.

These metrics describe the estimator only. End-to-end performance is established by the canonical LFM2.5 benchmark.

## Exactness contract

Every speculative method follows the same rule:

1. draft a block;
2. optionally apply its guidance/correction mechanism;
3. run the authoritative target verifier;
4. accept only the matching prefix;
5. insert the target token at the first mismatch;
6. continue until the requested output length is reached.

A higher acceptance rate, lower teacher error, lower guidance count or fewer target calls is **not** by itself a speed claim. Only measured wall-clock throughput from the same matched run supports a speedup statement.

See [`version12-parareal.md`](version12-parareal.md), [`version13-14.md`](version13-14.md), and [`reproducibility.md`](reproducibility.md).
