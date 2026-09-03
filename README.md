# DFlash Mini Lab

A reproducible **CPU-only speculative-decoding benchmark and visualization lab** for comparing five decoding mechanisms on the same tiny Transformer workload:

1. **Normal autoregressive decoding**
2. **DFlash-style parallel block speculative decoding**
3. **DFlash v2-style multi-candidate dynamic-programming path selection**
4. **DFlash3-MOBS** — experimental O(BK) middle-out bidirectional selection
5. **DFlash4-JUMP-MOBS** — experimental sparse indexed future-token anchors + O(BK) gap filling

> [!IMPORTANT]
> This repository is a **mechanism-level educational/reference implementation**, not the official DFlash/DFlash2 runtime or training recipe. DFlash3-MOBS and DFlash4-JUMP-MOBS are experiments introduced in this lab. Every speculative method uses the same target verifier, so final greedy output is checked against normal target-only decoding.

## Live benchmark

The GitHub Pages report is rebuilt from the CPU Docker benchmark and published only after exactness and artifact checks pass:

**https://vtavakkoli.github.io/DFlash-Mini-Lab/**

The report contains the latest throughput/latency matrix, acceptance rate, target-pass efficiency, measured selector/guidance work, animated architecture explanation and interactive charts.

## DFlash4-JUMP-MOBS in one picture

```text
current context h_t
      │
      ├──────────────► parallel drafter ─► top-k candidates at +1 +2 +3 +4
      │
      └──────────────► tiny jump head ───► approximate anchors at +2 and +4
                                              │
                            ┌─────────────────┘
                            ▼
                  lock sparse jump anchors
                            │
                  fill remaining gaps with
                  adjacent O(BK) scoring
                            │
                            ▼
                       TARGET VERIFY
                            │
                            ▼
                    exact greedy output
```

The jump head is separately trained from the same deterministic corpus. It predicts future tokens directly from the current context plus an offset embedding. The CPU reference block size is 4, so the sparse jump offsets are `+2` and `+4`.

## Why measure guidance work?

The lab does not assume a lower-complexity selector must be faster. DFlash2 evaluates an adjacent top-k transition grid with work proportional to **O(BK²)**. MOBS and JUMP-MOBS avoid the full `K × K` grid and keep local guidance proportional to **O(BK)** for a fixed sparse jump set.

JUMP-MOBS therefore reports separately:

- jump-head forward passes;
- jump-anchor candidate scores;
- local selector pair scores;
- total guidance scores;
- draft acceptance;
- tokens per target verification pass;
- end-to-end tokens/sec.

This lets the benchmark expose cases where proposal quality improves but the extra jump-head inference cost still outweighs the benefit.

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
  --jump-weight 0.5
```

Generated files:

```text
reports/
├── benchmark.json
├── report.html
├── speedup.gif
└── architecture.gif
```

## Local developer run

Docker is recommended because it deterministically rebuilds all reference weights, including the jump head. For a local run:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install numpy==2.3.5
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python training/build_weights.py --output-dir models
python -m pip install -e '.[test]'
pytest
CPU_THREADS=1 python -m dflash_mini_lab.cli --output-dir reports --top-k 8 --jump-weight 0.5
```

## Benchmark methodology

GitHub CI currently uses:

- fixed set of 8 prompts;
- 24 generated tokens per prompt;
- 1 warm-up iteration;
- 5 measured repeats;
- one CPU thread;
- one target SLM and tokenizer for every mode;
- speculative block size 4;
- top-k 8 for DFlash2/MOBS/JUMP-MOBS;
- JUMP offsets `+2,+4`;
- greedy decoding with exact-output verification.

> [!NOTE]
> The tiny target recomputes the visible sequence at every target call and intentionally does not implement a production KV cache. Absolute throughput should **not** be compared directly with llama.cpp/vLLM/SGLang serving numbers. The pipeline is for controlled relative algorithm study and reproducibility.

## Repository layout

```text
benchmarks/prompts.json          fixed workload
training/build_weights.py        deterministic target/drafter/selector/jump builder
models/                           generated weights/tokenizer (Docker build output)
src/dflash_mini_lab/runtime.py   NumPy target + speculative runtimes
src/dflash_mini_lab/decoding.py  five decoding modes + exact verifier correction
src/dflash_mini_lab/benchmark.py metrics + exactness + guidance-work collection
src/dflash_mini_lab/visuals.py   animated GIF generation
src/dflash_mini_lab/report.py    self-contained interactive HTML report
tests/                           exactness + complexity/artifact tests
.github/workflows/               reproducible CPU CI + GitHub Pages deployment
```

## References

- Jian Chen, Yesheng Liang, Zhijian Liu. **DFlash: Block Diffusion for Flash Speculative Decoding.** arXiv:2602.06036 (2026).
- Official DFlash project: https://github.com/z-lab/dflash
- vLLM Speculators documentation: https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/
- Inco AI: **DFlash 2: Keep Drafting Parallel** (2026): https://inco.ai/blog/dflash2/

See [`docs/algorithm.md`](docs/algorithm.md) for algorithm scope and [`docs/reproducibility.md`](docs/reproducibility.md) for benchmark controls.

## License

MIT.
