# DFlash Mini Lab

A reproducible **CPU-only benchmark and visualization lab** for comparing three LLM/SLM decoding mechanisms on the same tiny Transformer workload:

1. **Normal autoregressive decoding**
2. **DFlash-style parallel block speculative decoding**
3. **DFlash v2-style multi-candidate + predecessor-conditioned path selection**

The goal is to make the algorithms runnable, measurable and visually understandable without requiring a GPU or a multi-billion-parameter model.

> [!IMPORTANT]
> This repository is a **mechanism-level educational/reference implementation**, not the official DFlash/DFlash2 runtime or training recipe. Public production DFlash/DFlash2 checkpoints and serving integrations target much larger verifier models and accelerator-oriented engines. The lab retains the central parallel-draft → target-verify mechanics and lossless greedy verifier correction, while using a bundled tiny CPU model.

## What you get

- CPU-only Docker image; no CUDA/GPU runtime.
- Lightweight NumPy/BLAS float32 inference backend.
- Fixed tiny 2-layer causal Transformer SLM target.
- One-pass non-causal block drafter for DFlash mode.
- Learned low-rank top-k path selector for DFlash v2 mode.
- Identical prompts, token budget and verifier for all modes.
- Throughput, latency, speedup, acceptance and target-pass metrics.
- Automated exact-output tests.
- Animated speed-comparison GIF.
- Animated DFlash architecture GIF.
- Self-contained HTML report with embedded GIFs and interactive charts.
- GitHub Actions CPU benchmark artifact.

## Current sample result

The committed sample was generated on the development CPU with one BLAS thread and 24 generated tokens per prompt. It is included only to demonstrate the pipeline; run Docker on your machine for hardware-specific numbers.

| Mode | Median tok/s | Median latency | Speedup | Tokens / target pass | Draft acceptance | Exact vs normal |
|---|---:|---:|---:|---:|---:|:---:|
| Normal | 2409.40 | 9.96 ms | 1.00× | 1.00 | — | ✓ |
| DFlash | 4785.25 | 5.02 ms | 1.99× | 2.72 | 53.0% | ✓ |
| DFlash v2 | 4104.35 | 5.85 ms | 1.70× | 2.85 | 57.6% | ✓ |

A useful observation from this tiny CPU case is that DFlash v2 accepts more useful draft tokens per target pass, but its Python/NumPy path selector adds enough overhead that raw tok/s is below the simpler DFlash path. That distinction between **acceptance efficiency** and **end-to-end throughput** is one reason this lab exists.

## One-command Docker benchmark

```bash
docker build -t dflash-mini-lab .
docker run --rm \
  -e CPU_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 \
  -e OMP_NUM_THREADS=1 \
  -v "$PWD/reports:/app/reports" \
  dflash-mini-lab
```

Or:

```bash
docker compose run --rm benchmark
```

Generated files:

```text
reports/
├── benchmark.json
├── report.html
├── speedup.gif
└── architecture.gif
```

Open `reports/report.html` in a browser. The HTML is self-contained: both GIFs are embedded as base64 data and the interactive chart uses native JavaScript with no CDN dependency.

## Local Python run

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install -e '.[test]'
pytest
CPU_THREADS=1 python -m dflash_mini_lab.cli --output-dir reports
```

## Algorithm sketch

### Normal

```text
context → TARGET → t1 → TARGET → t2 → TARGET → t3 → ...
              one target pass for every generated token
```

### DFlash

```text
                     ┌─ d1
context → DRAFTER ───┼─ d2      all future slots drafted in parallel
       one pass      ├─ d3
                     └─ d4
                       │
                       ▼
               TARGET VERIFIER
                 one target pass
                       │
             accept matching prefix
             correct first mismatch
```

### DFlash v2

```text
parallel draft logits
       │
       ├─ top-k(position 1)
       ├─ top-k(position 2) ── learned predecessor-conditioned selector
       ├─ top-k(position 3)
       └─ top-k(position 4)
                    │
              selected path
                    │
              TARGET VERIFIER
```

The upstream DFlash2 design also adds local dynamic convolution around draft layers. This compact CPU lab focuses on the **multi-candidate/path-selection mechanism** requested for comparison and clearly reports that scope.

## Benchmark methodology

Default configuration:

- fixed set of 8 prompts;
- 24 generated tokens per prompt;
- 1 warm-up iteration;
- 3 measured repeats;
- one target SLM and tokenizer for every mode;
- block size 4 for speculative modes;
- top-k 4 for DFlash v2;
- greedy decoding for deterministic exactness checks;
- single-thread BLAS by default.

Metrics include median/mean generated tok/s, median/p95 decode latency, speedup vs normal, draft acceptance rate, target/draft forward-pass counts and generated tokens per target pass.

> [!NOTE]
> The tiny reference target recomputes the visible sequence at every target call and intentionally does not include a production KV cache. Absolute throughput therefore should **not** be compared with llama.cpp/vLLM/SGLang serving numbers. The pipeline is for controlled algorithm study and reproducibility.

## Repository layout

```text
benchmarks/prompts.json          fixed workload
models/tiny_dflash_lab.npz       float32 target/drafter/selector weights
models/tokenizer.json            fixed tokenizer
src/dflash_mini_lab/runtime.py   tiny NumPy Transformer + drafter runtime
src/dflash_mini_lab/decoding.py  normal / DFlash / DFlash v2 decoders
src/dflash_mini_lab/benchmark.py metrics + exactness collection
src/dflash_mini_lab/visuals.py   animated GIF generation
src/dflash_mini_lab/report.py    self-contained interactive HTML report
tests/                           exactness + artifact tests
.github/workflows/               reproducible CPU CI benchmark
```

## References

- Jian Chen, Yesheng Liang, Zhijian Liu. **DFlash: Block Diffusion for Flash Speculative Decoding.** arXiv:2602.06036 (2026).
- Official DFlash project: `z-lab/dflash`.
- vLLM Speculators documentation for DFlash and DFlash2.
- Inco AI: **DFlash 2: Keep Drafting Parallel** (2026).

See [`docs/algorithm.md`](docs/algorithm.md) for the exact scope of this implementation and [`docs/reproducibility.md`](docs/reproducibility.md) for benchmark controls.

## License

MIT.
