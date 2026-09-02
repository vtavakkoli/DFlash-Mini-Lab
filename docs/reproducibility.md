# Reproducibility guide

## Runtime contract

The default Docker image is CPU-only and intentionally excludes CUDA, GPU drivers, PyTorch and a training toolchain.
The Docker model-builder stage deterministically trains the tiny target, drafter and selector from source with seed 7 and pinned `torch==2.10.0` CPU wheels, then exports a float32 NPZ checkpoint. The final runtime stage contains only that generated checkpoint plus the NumPy/BLAS inference code; PyTorch is not present in the runtime image.

Pinned build/runtime dependencies:

- Builder: PyTorch 2.10.0 CPU
- Runtime: NumPy 2.3.5
- Runtime: Pillow 12.3.0

## Recommended benchmark settings

Use one BLAS thread for the most comparable single-process result:

```bash
docker build -t dflash-mini-lab .
docker run --rm \
  -e CPU_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 \
  -e OMP_NUM_THREADS=1 \
  -v "$PWD/reports:/app/reports" \
  dflash-mini-lab
```

The default workload uses:

- 8 fixed prompts;
- 24 generated tokens per prompt;
- 1 warm-up pass;
- 3 measured repeats;
- DFlash block size 4;
- DFlash v2 top-k 4.

## What the report records

`reports/benchmark.json` stores:

- mean and median generated tokens/sec;
- median and p95 end-to-end decode latency;
- speedup relative to normal decoding;
- target and draft forward-pass counts;
- draft acceptance rate;
- generated tokens per target pass;
- exact-output checks against normal greedy decoding;
- Python/NumPy/platform metadata.

## Benchmark interpretation

The bundled runtime intentionally recomputes the visible sequence at each target call and does not implement a production KV cache. This keeps the code short enough to inspect and keeps all three modes on the same reference backend, but it means absolute tok/s values must not be compared with llama.cpp, vLLM, SGLang or vendor benchmarks.

Use this repository for algorithm mechanics, regression testing and controlled relative comparisons. Use an upstream inference engine plus official compatible checkpoints for production serving claims.
