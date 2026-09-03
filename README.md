# DFlash Mini Lab

A reproducible **CPU-only speculative-decoding benchmark and visualization lab** comparing eight decoding mechanisms on the same tiny Transformer workload:

1. **Normal autoregressive decoding**
2. **DFlash-style** parallel block speculative decoding
3. **DFlash v2-style** multi-candidate dynamic-programming path selection
4. **DFlash3-MOBS** — experimental O(BK) middle-out bidirectional selection
5. **DFlash4-JUMP-MOBS** — sparse indexed future-token anchors + O(BK) gap filling
6. **DFlash5-FUSED-JUMP-MOBS** — shared-drafter sparse jump guidance with zero extra jump forward pass
7. **DFlash6-Boltzmann** — training-free deterministic Boltzmann/Gumbel exploration of existing draft candidates
8. **DFlash6-BMOBS** — one Boltzmann middle anchor followed by O(BK) MOBS gap filling

> [!IMPORTANT]
> This repository is a **mechanism-level educational/reference implementation**, not the official DFlash/DFlash2 runtime or training recipe. DFlash3 through DFlash6 are experiments introduced in this lab. Every speculative proposal is verified by the same target model and checked against normal greedy output.

## Live benchmark

The GitHub Pages report is rebuilt from the CPU Docker benchmark and published only after exactness and artifact checks pass:

**https://vtavakkoli.github.io/DFlash-Mini-Lab/**

## DFlash6 in one picture

```text
parallel DFlash draft logits
           │
           ▼
    top-k candidates
           │
   top-1/top-2 margin
           │
           ▼
 adaptive temperature
   high confidence → T↓
   uncertainty     → T↑
           │
      ┌────┴────┐
      │         │
      ▼         ▼
 Boltzmann    BMOBS
 Gumbel-Max   sample one
 every slot   middle anchor
      │         │
      │      O(BK) fill
      └────┬────┘
           ▼
      TARGET VERIFY
           ▼
   exact greedy output
```

DFlash6 adds **no model weights and no model forward pass**. Its Gumbel values are derived deterministically from the current context/token IDs, so CI remains reproducible rather than depending on nondeterministic RNG state.

### DFlash6-Boltzmann

The drafter's top-k candidates are sampled with deterministic Gumbel-Max. The base temperature is reduced automatically when the top-1/top-2 logit margin is large, making confident slots behave close to normal DFlash argmax.

### DFlash6-BMOBS

For even blocks, BMOBS chooses the more uncertain of the two center positions, samples that anchor with Boltzmann/Gumbel scoring, and fills the remaining positions with the existing linear MOBS neighbor scorer.

## What the first full DFlash6 run showed

On the initial 8 prompts × 24 tokens × 5 repeats PR run, the bounded sweep selected:

```text
DFlash6-Boltzmann temperature: 0.35
DFlash6-BMOBS temperature:     0.20
```

The same run measured:

- plain DFlash acceptance: **47.83%**;
- Boltzmann acceptance: **50.55%**;
- BMOBS acceptance: **53.18%**;
- Boltzmann and BMOBS remained exact after target verification;
- Boltzmann used no learned pair selector;
- BMOBS used about **81% less guidance work than DFlash2**;
- neither new mode beat plain DFlash in end-to-end CPU throughput because top-k/Gumbel/neighbor-selection overhead exceeded the saved verifier work.

This negative throughput result is preserved rather than tuning until a runner produces a desired ranking.

## Bounded optimization

CI performs bounded, reproducible searches rather than open-ended tuning:

- DFlash5: fixed fused-weight / margin candidates;
- DFlash6-Boltzmann: temperatures `0.02, 0.05, 0.10, 0.20, 0.35`;
- DFlash6-BMOBS: the same temperature set, optimized independently.

The selected settings are then evaluated on the full **8 prompts × 24 generated tokens × 5 repeats** workload. CI artifacts include `dflash5-sweep.json` and `dflash6-sweep.json`.

## One-command Docker benchmark

```bash
docker build -t dflash-mini-lab .
docker run --rm \
  -e CPU_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 \
  -e OMP_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  -v "$PWD/reports:/app/reports" \
  dflash-mini-lab \
  --top-k 8 \
  --mobs-refine-passes 0 \
  --jump-weight 0.5 \
  --fused-jump-weight 1.0 \
  --boltzmann-temp 0.35 \
  --bmobs-temp 0.20
```

Generated files:

```text
reports/
├── benchmark.json
├── report.html
├── speedup.gif
└── architecture.gif
```

## Benchmark methodology

GitHub CI uses one CPU/BLAS thread, 8 fixed prompts, 24 generated tokens per prompt, one warm-up, five measured repeats, block size 4 and top-k 8. Every speculative method is verified against the same target-only greedy reference.

The tiny target intentionally recomputes the visible sequence and has no production KV cache. Absolute throughput should therefore **not** be compared directly with llama.cpp/vLLM/SGLang production serving numbers.

## Repository layout

```text
benchmarks/prompts.json                    fixed workload
training/build_weights.py                  deterministic model/speculator builder
src/dflash_mini_lab/runtime.py             NumPy target + learned selectors
src/dflash_mini_lab/boltzmann.py           deterministic DFlash6 sampling/selectors
src/dflash_mini_lab/optimize_boltzmann.py  bounded DFlash6 temperature sweep
src/dflash_mini_lab/decoding.py            eight decoding modes + exact correction
src/dflash_mini_lab/benchmark.py           metrics + exactness + guidance work
src/dflash_mini_lab/visuals.py             animated GIF generation
src/dflash_mini_lab/report.py              self-contained interactive HTML report
tests/                                     exactness + complexity/artifact tests
.github/workflows/                         optimization + CPU CI + Pages deployment
```

## References

- Jian Chen, Yesheng Liang, Zhijian Liu. **DFlash: Block Diffusion for Flash Speculative Decoding.** arXiv:2602.06036 (2026).
- Official DFlash project: https://github.com/z-lab/dflash
- vLLM Speculators documentation: https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/
- Inco AI: **DFlash 2: Keep Drafting Parallel** (2026): https://inco.ai/blog/dflash2/

See [`docs/algorithm.md`](docs/algorithm.md) for implementation scope.

## License

MIT.
