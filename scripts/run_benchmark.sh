#!/usr/bin/env sh
set -eu

: "${CPU_THREADS:=1}"
export CPU_THREADS
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"

python -m dflash_mini_lab.cli \
  --output-dir "${OUTPUT_DIR:-reports}" \
  --tokens "${TOKENS:-24}" \
  --warmups "${WARMUPS:-1}" \
  --repeats "${REPEATS:-3}" \
  --top-k "${TOP_K:-4}"
