# Version 8 — EAGLE-3 baseline

Version 8 adds a ready-checkpoint **EAGLE-3** baseline to DFlash Mini Lab and measures the complete generation call rather than only target-model call reduction.

## Matched model pair

- target: `Qwen/Qwen3-1.7B`
- draft: `AngelSlim/Qwen3-1.7B_eagle3`
- SafeAILab/EAGLE source: `cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b`
- EAGLE tree budget: 60
- depth: 7
- draft top-k: 10

The public EAGLE checkpoint is paired with `Qwen/Qwen3-1.7B`, while the earlier DFlash 1.7B study in this repository uses `Qwen/Qwen3-1.7B-Base`. For that reason, compare **speedup ratios against each method's matched normal target**, not raw tok/s across the two studies.

## What is timed

The benchmark times the whole `naivegenerate` or `eagenerate` call after one warm-up of each path. The reported wall time therefore includes:

- target prefill;
- EAGLE draft-tree generation;
- target tree verification;
- KV/cache maintenance;
- accepted-tree-path processing;
- Python/runtime overhead.

No model-loading or first-use cache-allocation time is included in the measured generation calls.

## Verified CPU result

GitHub Actions configuration: Ubuntu runner, 2 CPU threads, float32, 6 held-out prompts, 16 requested greedy tokens per prompt, one measured repeat.

| Method | Median tok/s | vs matched normal | Median wall time | Mean output tokens / iteration | Exact |
|---|---:|---:|---:|---:|:---:|
| Normal Qwen in EAGLE runtime | **2.995** | **1.000×** | 5.344 s | 1.00 | ✓ |
| **Version 8 — EAGLE-3** | **0.680** | **0.227×** | 23.534 s | **1.84** | ✓ |

Model/runtime parameter counts reported by the benchmark:

- target: **1,720,574,976** parameters;
- EAGLE runtime draft module: **448,012,288** parameters.

EAGLE-3 clearly improves sequential efficiency in this run—about 1.84 requested output tokens per tree iteration—but the 60-node tree and draft/verification work are too expensive on a two-thread CPU. The ready method is therefore **not a CPU speedup** in this environment.

This does not contradict published GPU-oriented EAGLE results. Speculative tree verification is designed to exchange additional parallel compute for fewer sequential target steps. That trade can be favorable on under-utilized GPU execution and unfavorable on CPU.

## Exactness and dtype

The first diagnostic run used bf16 and observed at least one greedy-prefix divergence between sequential and tree evaluation. As with the earlier Qwen DFlash experiment, batching/tree evaluation can move a near-tied argmax under reduced precision.

The authoritative Version 8 CPU result therefore uses **float32**. All six requested output prefixes matched the same target model's `naivegenerate` output token-for-token, and CI fails if any mismatch is present.

## Reproduce

```bash
docker build -f Dockerfile.eagle3 -t dflash-eagle3-v8 .

docker run --rm \
  -e HF_HOME=/cache \
  -e CPU_THREADS=2 \
  -e OMP_NUM_THREADS=2 \
  -e MKL_NUM_THREADS=2 \
  -e OPENBLAS_NUM_THREADS=2 \
  -v "$PWD/hf-eagle3-cache:/cache" \
  -v "$PWD/eagle3-reports:/app/eagle3-reports" \
  dflash-eagle3-v8 \
    --base-model Qwen/Qwen3-1.7B \
    --eagle-model AngelSlim/Qwen3-1.7B_eagle3 \
    --prompts /app/qwen_benchmarks/prompts.json \
    --output-dir /app/eagle3-reports \
    --prompt-limit 6 \
    --tokens 16 \
    --repeats 1 \
    --dtype float32 \
    --cpu-threads 2 \
    --total-token 60 \
    --depth 7 \
    --draft-top-k 10 \
    --threshold 1.0 \
    --max-length 256
```

Target and draft weights are downloaded at runtime and are not redistributed by DFlash Mini Lab.
