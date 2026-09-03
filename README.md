# DFlash Mini Lab

A reproducible **CPU-only speculative-decoding benchmark and visualization lab**. The repository contains the original eight-way tiny reference benchmark plus separate real-model paths for LFM2.5 and **Qwen3-0.6B-Base**.

The tiny reference workload compares:

1. **Normal autoregressive decoding**
2. **DFlash-style** parallel block speculative decoding
3. **DFlash v2-style** multi-candidate dynamic-programming path selection
4. **DFlash3-MOBS** — experimental O(BK) middle-out bidirectional selection
5. **DFlash4-JUMP-MOBS** — sparse indexed future-token anchors + O(BK) gap filling
6. **DFlash5-FUSED-JUMP-MOBS** — shared-drafter sparse jump guidance with zero extra jump forward pass
7. **DFlash6-Boltzmann** — training-free deterministic Boltzmann/Gumbel exploration of existing draft candidates
8. **DFlash6-BMOBS** — one Boltzmann middle anchor followed by O(BK) MOBS gap filling

The Qwen path additionally evaluates **DFlash7-ACT**, an adaptive cost-aware speculative router on a real cached verifier.

> [!IMPORTANT]
> This repository is a **mechanism-level educational/reference implementation**, not the official DFlash/DFlash2 runtime or training recipe. DFlash3 through DFlash7 are experiments introduced in this lab. Every speculative proposal is verified by the same target model and checked against normal greedy output.

## Live tiny benchmark

The GitHub Pages report is rebuilt from the tiny CPU Docker benchmark and published only after exactness and artifact checks pass:

**https://vtavakkoli.github.io/DFlash-Mini-Lab/**

## Real Qwen3-0.6B benchmark — DFlash7-ACT

The Qwen benchmark fixes several fidelity/runtime limitations of the earlier tiny/LFM paths:

```text
Qwen3-0.6B-Base verifier
        │
        ├── DynamicCache KV state
        ├── hidden layer 4
        ├── hidden layer 14
        └── hidden layer 27
                 │
                 ▼
        learned hidden fusion
                 │
       verifier memory as K/V
                 │
                 ▼
 known greedy anchor + 5 mask slots
                 │
       2-layer bidirectional
          block drafter
                 │
                 ▼
        retained target-head rows
                 │
                 ▼
           draft candidates
                 │
          ┌──────┴──────┐
          │             │
          ▼             ▼
 fixed DFlash      DFlash7-ACT
 full suffix       margin-gated suffix
          │             │
          └──────┬──────┘
                 ▼
       one Qwen verifier call
                 │
         accept exact prefix
                 │
                 ▼
      DynamicCache.crop(reject)
```

The **anchor is not drafted**. It is the target verifier's already-known greedy bonus token from the previous logits. The drafter predicts only tokens after that exact anchor. Verification forwards `[anchor + draft suffix]` in one Qwen call, then crops rejected cache entries.

### DFlash7-ACT

**ACT = Adaptive Cost-aware Token speculation.** DFlash7 uses the already-computed top-1/top-2 draft-logit margin to shorten an uncertain suffix before it reaches the expensive verifier. It adds no neural forward pass. CI calibrates a bounded threshold set on separate prompts:

```text
0.00, 0.25, 0.50, 1.00, 1.50, 2.00
```

Threshold `0` is full fixed-block DFlash, so ACT can fall back to the baseline.

### First verified Qwen result

Configuration:

- target: `Qwen/Qwen3-0.6B-Base`, **596,049,920 parameters**;
- target hidden size 1024, vocabulary 151,936;
- retained draft vocabulary: **3,251 tokens**;
- held-out continuation candidate coverage: **93.94%**;
- hidden layers: 4 / 14 / 27;
- hidden-memory window: 16 verifier tokens;
- block size 6 = 1 exact anchor + 5 parallel draft slots;
- drafter: 2 decoder layers, width 256, 8 heads, about **3.16M parameters**;
- training: 24 Qwen teacher seeds × 18 generated tokens, 312 resulting block examples;
- benchmark: 6 held-out prompts × 12 tokens × 2 repeats, float32 CPU / 2 threads;
- decode timing excludes prefill.

Measured PR run:

| Method | Median tok/s | vs normal | Draft acceptance | Mean target calls | Mean target input tokens | Exact |
|---|---:|---:|---:|---:|---:|:---:|
| Normal cached Qwen | **7.468** | **1.000×** | — | 11.00 | 11.00 | ✓ |
| Hidden-fusion fixed DFlash | 3.634 | 0.487× | 2.81% | **10.17** | 52.33 | ✓ |
| **DFlash7-ACT** | **7.000** | **0.937×** | 0.00% at selected policy | 11.00 | 11.83 | ✓ |

The bounded ACT calibration selected margin threshold **2.0**. On calibration prompts, full-block DFlash (`T=0`) measured about 3.994 tok/s, while the selected ACT policy measured about 7.269 tok/s by suppressing nearly all unprofitable speculative suffixes.

This is an intentionally preserved negative end-to-end result: **DFlash7 does not beat normal cached Qwen in this small CPU experiment.** It does, however, recover most of the severe fixed-DFlash penalty by identifying that the current drafter is not profitable on held-out prompts.

The key diagnostic is generalization. The drafter's training-position accuracies were approximately **43.6%, 25.3%, 19.9%, 17.3%, 16.3%**, while held-out accepted-prefix rate was much lower. With only 312 Qwen teacher examples, the bottleneck is now drafter data/generalization rather than missing anchor or KV-cache semantics.

> [!NOTE]
> This Qwen drafter is substantially closer to DFlash than the original educational path, but it is still not an official DFlash checkpoint or exact upstream recipe. It uses three selected target layers, a 16-token hidden-memory window, a 2-layer cross-attention decoder, a reduced candidate vocabulary and a D-PACE-inspired loss rather than the complete upstream training stack.

### Run the Qwen benchmark

```bash
docker build -f Dockerfile.qwen -t dflash-qwen7 .

# 1. Build the small hidden-fusion drafter from the frozen Qwen target.
docker run --rm \
  -v "$PWD/hf-qwen-cache:/cache" \
  -v "$PWD/qwen-artifacts:/app/qwen-artifacts" \
  dflash-qwen7 \
  python -m dflash_mini_lab.qwen_prepare \
    --model-id Qwen/Qwen3-0.6B-Base \
    --output-dir /app/qwen-artifacts

# 2. Benchmark normal cached Qwen, fixed DFlash and DFlash7-ACT.
docker run --rm \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HUB_OFFLINE=1 \
  -v "$PWD/hf-qwen-cache:/cache" \
  -v "$PWD/qwen-artifacts:/app/qwen-artifacts:ro" \
  -v "$PWD/qwen-reports:/app/qwen-reports" \
  dflash-qwen7 \
  python -m dflash_mini_lab.qwen_cli \
    --aux /app/qwen-artifacts/qwen_dflash.pt \
    --output-dir /app/qwen-reports
```

The target weights are downloaded from Hugging Face and are **not** committed or redistributed by this repository.

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
- DFlash6-BMOBS: the same temperature set, optimized independently;
- DFlash7-ACT: verifier-suffix margin thresholds `0, 0.25, 0.5, 1, 1.5, 2`, calibrated on separate Qwen prompts.

## One-command tiny Docker benchmark

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

## Benchmark methodology

The tiny GitHub CI path uses one CPU/BLAS thread, 8 fixed prompts, 24 generated tokens per prompt, one warm-up, five measured repeats, block size 4 and top-k 8. Every speculative method is verified against the same target-only greedy reference.

The tiny target intentionally recomputes the visible sequence and has no production KV cache. Its absolute throughput should therefore **not** be compared directly with llama.cpp/vLLM/SGLang production serving numbers. The Qwen path is separate and uses a real DynamicCache verifier.

## Repository layout

```text
benchmarks/prompts.json                    tiny fixed workload
training/build_weights.py                  deterministic tiny model/speculator builder
src/dflash_mini_lab/runtime.py             NumPy tiny target + learned selectors
src/dflash_mini_lab/boltzmann.py           deterministic DFlash6 sampling/selectors
src/dflash_mini_lab/decoding.py            tiny eight-way decoding + exact correction
src/dflash_mini_lab/qwen_aux.py            hidden-fusion Qwen block drafter
src/dflash_mini_lab/qwen_prepare.py        Qwen teacher generation + drafter training
src/dflash_mini_lab/qwen_runtime.py        cached Qwen verifier + rollback
src/dflash_mini_lab/qwen_benchmark.py      Normal / DFlash / DFlash7 benchmark
qwen_benchmarks/                           Qwen train, calibration and test prompts
Dockerfile.qwen                            reproducible Qwen CPU environment
.github/workflows/qwen-dflash7.yml         real Qwen exactness/performance CI
```

## References

- Jian Chen, Yesheng Liang, Zhijian Liu. **DFlash: Block Diffusion for Flash Speculative Decoding.** arXiv:2602.06036 (2026).
- Official DFlash project: https://github.com/z-lab/dflash
- vLLM Speculators documentation: https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/
- NVIDIA Model Optimizer DFlash documentation/examples.
- Qwen3 model family: `Qwen/Qwen3-0.6B-Base`.

See [`docs/algorithm.md`](docs/algorithm.md) for implementation scope.

## License

MIT.
