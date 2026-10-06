#!/bin/bash
cd "$(dirname "$0")/../Mind2Web"
python download_data.py
python run.py --model ccli-sonnet --memory-mode efm \
    --memory-dir memories_efm_cmp --output-root results_efm_cmp --limit 80 --efm-n-inject 3 "$@"
