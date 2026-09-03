# DFlash Mini Lab

A reproducible **CPU-only speculative-decoding benchmark and visualization lab** comparing six decoding mechanisms on the same tiny Transformer workload:

1. **Normal autoregressive decoding**
2. **DFlash-style** parallel block speculative decoding
3. **DFlash v2-style** multi-candidate dynamic-programming path selection
4. **DFlash3-MOBS** — experimental O(BK) middle-out bidirectional selection
5. **DFlash4-JUMP-MOBS** — experimental sparse indexed future-token anchors + O(BK) gap filling
6. **DFlash5-FUSED-JUMP-MOBS** — experimental shared-drafter, candidate-only sparse jump guidance with **zero extra jump forward pass**

> [!IMPORTANT]
> This repository is a **mechanism-level educational/reference implementation**, not the official DFlash/DFlash2 runtime or training recipe. DFlash3, DFlash4 and DFlash5 are experiments introduced in this lab. Every speculative path is verified by the same target model and checked against normal greedy output.

## Live benchmark

The GitHub Pages report is rebuilt from the CPU Docker benchmark and published only after exactness and artifact checks pass:

**https://vtavakkoli.github.io/DFlash-Mini-Lab/**

The page includes the six-way throughput/latency matrix, acceptance, target-pass efficiency, measured guidance work, animated architecture and interactive charts.

## DFlash5 in one picture

```text
current context h_t
      │
      ▼
parallel drafter ─────────────── one forward pass
      │
      ├── normal logits ───────► top-k candidates at +1 +2 +3 +4
      │
      └── shared hidden states
                  │
                  ▼
          low-rank fused residual
          only for top-k at +2/+4
                  │
                  ▼
          confidence-gated anchors
                  │
          O(BK) local gap filling
                  │
                  ▼
              TARGET VERIFY
                  │
                  ▼
          exact greedy output
```

DFlash5 specifically targets the bottleneck measured in DFlash4: DFlash4 improves speculative path quality but pays for a **separate jump-head forward pass**. DFlash5 trains a small low-rank residual from the already-computed drafter states and evaluates only retained candidates. The DFlash4 head can be used as a teacher during deterministic model building, but it is absent from DFlash5 inference.

## Optimization strategy

CI does not assume one fused weight is best. After building the deterministic Docker image once, it runs a bounded sweep of fused residual weight / confidence-margin settings on a short benchmark. It selects the fastest configuration that still satisfies:

- exact output equivalence;
- zero DFlash5 jump-network forward passes;
- non-zero fused anchor use;
- total DFlash5 guidance work below DFlash2.

The selected configuration is then evaluated on the full workload: **8 prompts × 24 generated tokens × 5 measured repeats**. The sweep is saved as `dflash5-sweep.json` in the CI artifact and on GitHub Pages.

The optimization history is intentionally preserved: removing duplicate top-k selection materially reduced DFlash5 overhead, while a later teacher-distillation experiment did not consistently improve acceptance. The repository reports those outcomes rather than tuning until one runner produces a desired ranking.

## Why measure guidance work?

DFlash2 constructs adjacent top-k transition grids with selector work proportional to **O(BK²)**. MOBS and the jump variants avoid a full `K × K` grid.

For fixed sparse anchor count `J` and low-rank width `R`, DFlash5 guidance is approximately:

```text
candidate residual: O(JKR)
gap filling:        O(BK)
```

The key runtime property is that the residual consumes **existing drafter states**; it does not add another Transformer/MLP forward pass.

Lower guidance complexity alone does not guarantee higher tokens/sec. A cheaper selector can produce a weaker proposal and cause more target verification. The benchmark therefore reports both computation and proposal quality.

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
  --fused-jump-min-margin 0.0
```

Generated files:

```text
reports/
├── benchmark.json
├── report.html
├── speedup.gif
└── architecture.gif
```

CI additionally publishes:

```text
dflash5-sweep.json
```

## Local developer run

Docker is recommended because it deterministically rebuilds all reference weights.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install numpy==2.3.5
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python training/build_weights.py --output-dir models
python -m pip install -e '.[test]'
pytest
CPU_THREADS=1 python -m dflash_mini_lab.cli \
  --output-dir reports --top-k 8 \
  --jump-weight 0.5 \
  --fused-jump-weight 1.0
```

## Benchmark methodology

GitHub CI uses:

- 8 fixed prompts;
- 24 generated tokens per prompt for the full run;
- 1 warm-up iteration;
- 5 measured repeats;
- one CPU/BLAS thread;
- one target SLM and tokenizer for every method;
- speculative block size 4;
- top-k 8 for DFlash2/MOBS/JUMP variants;
- sparse offsets `+2,+4`;
- greedy decoding with exact target verification.

> [!NOTE]
> The tiny target recomputes the visible sequence at every target call and intentionally does not implement a production KV cache. Absolute throughput should **not** be compared directly with llama.cpp/vLLM/SGLang serving numbers. This pipeline is for controlled relative algorithm study and reproducibility.

## Repository layout

```text
benchmarks/prompts.json          fixed workload
training/build_weights.py        deterministic target/drafter/selector/jump/fused builder
models/                           generated weights/tokenizer (Docker build output)
src/dflash_mini_lab/runtime.py   NumPy target + speculative runtimes
src/dflash_mini_lab/decoding.py  six decoding modes + exact verifier correction
src/dflash_mini_lab/benchmark.py metrics + exactness + guidance-work collection
src/dflash_mini_lab/visuals.py   animated GIF generation
src/dflash_mini_lab/report.py    self-contained interactive HTML report
tests/                           exactness + complexity/artifact tests
.github/workflows/               bounded optimization + CPU CI + GitHub Pages deployment
```

## References

- Jian Chen, Yesheng Liang, Zhijian Liu. **DFlash: Block Diffusion for Flash Speculative Decoding.** arXiv:2602.06036 (2026).
- Official DFlash project: https://github.com/z-lab/dflash
- vLLM Speculators documentation: https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/
- Inco AI: **DFlash 2: Keep Drafting Parallel** (2026): https://inco.ai/blog/dflash2/

See [`docs/algorithm.md`](docs/algorithm.md) for algorithm scope and [`docs/reproducibility.md`](docs/reproducibility.md) for benchmark controls.

## License

MIT.
