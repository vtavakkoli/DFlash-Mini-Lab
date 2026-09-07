#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_DIR="${ARTIFACT_DIR:-/app/lfm-artifacts}"
REPORT_DIR="${REPORT_DIR:-/app/gpu-reports}"
SEEDS="/app/real_benchmarks/train_seeds.json"
PROMPTS="/app/real_benchmarks/prompts.json"
CALIBRATION="/app/real_benchmarks/calibration_prompts.json"
CPU_THREADS="${CPU_THREADS:-4}"

mkdir -p "$ARTIFACT_DIR" "$REPORT_DIR" /cache

echo "============================================================"
echo " DFlash Mini Lab · GPU capability check"
echo "============================================================"
python -m dflash_mini_lab.lfm_gpu_check

echo
echo "GPU check passed. Persistent artifacts: $ARTIFACT_DIR"
echo "Target/drafter benchmark dtype: ${LFM_GPU_DTYPE:-float16}"

if [[ ! -s "$ARTIFACT_DIR/lfm_aux.pt" ]]; then
  echo
echo "[1/4] Preparing frozen LFM2.5 DFlash backbone (first run only)..."
  python -m dflash_mini_lab.lfm_prepare \
    --model-id LiquidAI/LFM2.5-350M-Base \
    --output-dir "$ARTIFACT_DIR" \
    --max-seed-count 40 --generation-tokens 32 \
    --top-candidate-k 32 --candidate-limit 2048 \
    --drafter-steps 300 --selector-steps 180 \
    --jump-steps 120 --fused-steps 180 --cpu-threads "$CPU_THREADS"
else
  echo "[1/4] Reusing $ARTIFACT_DIR/lfm_aux.pt"
fi

if [[ ! -s "$ARTIFACT_DIR/lfm_dspark.pt" ]]; then
  echo
echo "[2/4] Preparing V9 DSpark-Lite heads (first run only)..."
  python -m dflash_mini_lab.lfm_dspark \
    --aux "$ARTIFACT_DIR/lfm_aux.pt" \
    --output "$ARTIFACT_DIR/lfm_dspark.pt" \
    --seeds "$SEEDS" \
    --max-seed-count 40 --generation-tokens 32 \
    --markov-rank 16 --markov-steps 180 \
    --confidence-steps 100 --batch-size 64 --cpu-threads "$CPU_THREADS"
else
  echo "[2/4] Reusing $ARTIFACT_DIR/lfm_dspark.pt"
fi

if [[ ! -s "$ARTIFACT_DIR/v12_parareal.json" ]]; then
  echo
echo "[3/4] Preparing V12 PARAREAL model (first run only)..."
  python -m dflash_mini_lab.v12_prepare \
    --aux "$ARTIFACT_DIR/lfm_aux.pt" \
    --output "$ARTIFACT_DIR/v12_parareal.json" \
    --seeds "$SEEDS" \
    --max-seed-count 24 --generation-tokens 24 \
    --top-k 8 --correction-rounds 2 --damping 0.75 \
    --holdout-fraction 0.20 --cpu-threads "$CPU_THREADS"
else
  echo "[3/4] Reusing $ARTIFACT_DIR/v12_parareal.json"
fi

if [[ ! -s "$ARTIFACT_DIR/v14_simple_parareal.json" ]]; then
  echo
echo "[4/4] Preparing V14 Simple PARAREAL estimator (first run only)..."
  python -m dflash_mini_lab.v14_prepare \
    --aux "$ARTIFACT_DIR/lfm_aux.pt" \
    --output "$ARTIFACT_DIR/v14_simple_parareal.json" \
    --seeds "$SEEDS" \
    --max-seed-count 24 --generation-tokens 24 \
    --ridge 0.001 --damping 1.0 --switch-threshold 0.0 \
    --max-corrections 1 --holdout-fraction 0.20 --cpu-threads "$CPU_THREADS"
else
  echo "[4/4] Reusing $ARTIFACT_DIR/v14_simple_parareal.json"
fi

echo
echo "============================================================"
echo " LFM2.5 All-14 CUDA benchmark"
echo "============================================================"
python -m dflash_mini_lab.lfm_gpu_benchmark \
  --aux "$ARTIFACT_DIR/lfm_aux.pt" \
  --dspark "$ARTIFACT_DIR/lfm_dspark.pt" \
  --v12-model "$ARTIFACT_DIR/v12_parareal.json" \
  --v14-model "$ARTIFACT_DIR/v14_simple_parareal.json" \
  --prompts "$PROMPTS" \
  --calibration-prompts "$CALIBRATION" \
  --output-dir "$REPORT_DIR" \
  --tokens "${GPU_TOKENS:-24}" \
  --repeats "${GPU_REPEATS:-3}" \
  --prompt-limit "${GPU_PROMPT_LIMIT:-6}" \
  --calibration-tokens "${GPU_CALIBRATION_TOKENS:-8}" \
  --calibration-prompt-limit "${GPU_CALIBRATION_PROMPT_LIMIT:-3}" \
  --top-k 8 --jump-weight 0.5 --fused-weight 1.0 \
  --boltzmann-temperature 0.15 --bmobs-temperature 0.35 \
  --cpu-threads "$CPU_THREADS" \
  --dtype "${LFM_GPU_DTYPE:-float16}"

echo
echo "GPU benchmark complete."
echo "JSON:   $REPORT_DIR/benchmark.json"
echo "Report: $REPORT_DIR/index.html"
