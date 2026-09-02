#!/usr/bin/env sh
set -eu
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python training/build_weights.py --output-dir models
