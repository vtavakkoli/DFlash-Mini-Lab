# Shared LFM inference optimizations

The optimization applies to the canonical LFM2.5 target and all fourteen lab
methods, on CPU and CUDA. It reuses the prepared model weights; no retraining
or quantization is required.

## Changes

- **Only project the needed suffix.** Normal decoding needs one vocabulary row.
  Verifying B draft tokens needs B+1 rows, including the bonus prediction.
  The target's `logits_to_keep` argument avoids running the vocabulary output
  layer on all earlier hidden states. Models without that argument retain full
  projection. The backbone still processes the complete prefix.
- **Reduce on the target device.** Torch selects greedy token IDs before the
  CPU transfer. CUDA returns B+1 integer IDs instead of a sequence-by-vocabulary
  float32 array, and the transfer synchronizes the measured target call.
- **Commit the bonus token.** If every draft matches, the last target row gives
  one extra greedy token without another forward pass. On rejection, only the
  accepted prefix and the first correction are committed. Output length is
  still bounded by the requested token budget.
- **Optional width tuning.** `--tune` tests widths 1, 2 and 4 (bounded by the
  artifact block size), warms each candidate, rotates evaluation order, and
  chooses each method's fastest exact configuration by total decode time.
  It runs only on calibration prompts; overlap with benchmark prompts is an
  error. ACT/DSpark confidence gates may shorten the chosen cap further.

Normal decoding receives the same target projection/argmax optimization.
The independent exactness oracle retains the original full-logit computation.
An exactness failure saves diagnostic output and returns a failing exit status.

## Run on your machine

```bash
python run_cpu.py --tune
python run_gpu.py --tune
```

For a smaller first check:

```bash
python run_cpu.py --smoke --tune
python run_gpu.py --smoke --tune
```

`--tuning-repeats 2` is the default. Tuning incurs a separate startup cost and
does not guarantee that its calibration winner is fastest on every workload.
Use the resulting held-out timings to assess the improvement.

Docker supports the same options:

```bash
docker compose --profile cpu run --rm test-cpu python run_cpu.py --tune
docker compose --profile gpu run --rm test-gpu python run_gpu.py --tune --artifact-dir /app/lfm-artifacts
```

## Reproduce a matched before/after comparison

After creating a tuned report, run:

```bash
python scripts/benchmark_optimizations.py --settings cpu-reports/benchmark.json
```

For CUDA, add `--gpu --artifact-dir gpu-artifacts`, point `--settings` at the
GPU report and `--output` at `gpu-reports/optimization-comparison.json`. Match
the CPU thread count and target dtype used for calibration.

This script interleaves three modes on one loaded model with identical frozen
selector configurations and held-out prompts: a legacy replay (full logits,
no bonus, full width), the optimized runtime at full width, and the optimized
runtime at calibrated widths. It warms every mode/method combination and saves
raw timings, exactness, throughput and paired per-method speedups. The legacy
mode replays the old behavior through the shared verifier, not a separate
checkout. Timings exclude loading, training, calibration and warmup.

For individual ablations, set `LFM_OPTIMIZE_TARGET=0` to restore full-logit
transfer/reduction or `LFM_BONUS_TOKEN=0` to disable bonus-token commits. Omit
`--tune` for full verification width. The report records all three settings.

## Remaining work

The target still uses `use_cache=False`. LFM has both attention and recurrent
convolution state; a rejected speculative suffix requires restoring both. A
plain KV-cache crop is insufficient. Implementing and validating hybrid cache
rollback is a separate optimization, and current speedups must not be presented
as comparisons against an optimized cached serving engine.

Tests use a real, randomly initialized tiny hybrid LFM and all actual auxiliary
modules without downloading weights. Controlled full-acceptance and rejection
tests cover the bonus/correction logic. CUDA dtype tests run when a compatible
GPU is available; a CPU-only test run does not establish GPU speed or exactness.

```bash
python -m pytest -q
```
