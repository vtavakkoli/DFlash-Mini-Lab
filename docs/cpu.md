# LFM2.5 All-14 CPU benchmark

The CPU path mirrors the GPU interface and keeps CPU/GPU evidence separate.

## Docker Compose

Run the canonical CPU benchmark with:

```bash
docker compose --profile cpu run --rm test-cpu
```

The service:

1. uses the CPU-only PyTorch 2.10 runtime;
2. does not install the repository package;
3. downloads the same prepared LFM2.5 auxiliary artifacts used by the GPU path;
4. verifies every artifact by SHA-256 and byte size;
5. runs Normal + all 14 speculative methods in float32;
6. preserves the exact greedy-output gate;
7. writes a separate CPU report.

## Outputs

```text
cpu-reports/index.html
cpu-reports/report.html
cpu-reports/benchmark.json
```

Downloaded artifacts are cached in:

```text
cpu-artifacts/
```

The target model itself is still loaded from `LiquidAI/LFM2.5-350M-Base` through the Hugging Face cache mounted at `hf-cache/`.

## Workload controls

Defaults:

```text
CPU_THREADS=2
CPU_TOKENS=24
CPU_REPEATS=3
CPU_PROMPT_LIMIT=6
CPU_CALIBRATION_TOKENS=8
CPU_CALIBRATION_PROMPT_LIMIT=3
```

PowerShell example for a smaller run:

```powershell
$env:CPU_THREADS="4"
$env:CPU_TOKENS="8"
$env:CPU_REPEATS="1"
$env:CPU_PROMPT_LIMIT="1"
docker compose --profile cpu run --rm test-cpu
```

## Direct Python

A fresh clone can also run without package installation:

```bash
python run_cpu.py
```

Quick smoke test:

```bash
python run_cpu.py --smoke
```

Download artifacts only:

```bash
python run_cpu.py --download-only
```

The direct runner installs only missing runtime dependencies when needed. It never runs `pip install .` or `pip install -e .`.

## Comparing CPU and GPU

Use the two matched surfaces independently:

```bash
docker compose --profile cpu run --rm test-cpu
docker compose --profile gpu run --rm test-gpu
```

CPU output lives in `cpu-reports/`; GPU output lives in `gpu-reports/`. Compare absolute throughput only when tokens, prompts, repeats, artifacts, and other benchmark settings are matched.
