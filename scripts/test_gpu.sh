#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_DIR="${ARTIFACT_DIR:-/app/lfm-artifacts}"
REPORT_DIR="${REPORT_DIR:-/app/gpu-reports}"
CPU_THREADS="${CPU_THREADS:-4}"

mkdir -p "$ARTIFACT_DIR" "$REPORT_DIR" /cache

echo "============================================================"
echo " DFlash Mini Lab · direct GPU runner"
echo "============================================================"
echo "Artifacts: $ARTIFACT_DIR"
echo "Reports:   $REPORT_DIR"
echo "Dtype:     ${LFM_GPU_DTYPE:-float16}"
echo

python /app/run_gpu.py \
  --artifact-dir "$ARTIFACT_DIR" \
  --output-dir "$REPORT_DIR" \
  --tokens "${GPU_TOKENS:-24}" \
  --repeats "${GPU_REPEATS:-3}" \
  --prompt-limit "${GPU_PROMPT_LIMIT:-6}" \
  --calibration-tokens "${GPU_CALIBRATION_TOKENS:-8}" \
  --calibration-prompt-limit "${GPU_CALIBRATION_PROMPT_LIMIT:-3}" \
  --cpu-threads "$CPU_THREADS" \
  --dtype "${LFM_GPU_DTYPE:-float16}" \
  --no-bootstrap

echo
echo "GPU benchmark complete."
echo "JSON:   $REPORT_DIR/benchmark.json"
echo "Report: $REPORT_DIR/index.html"
