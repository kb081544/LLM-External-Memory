#!/bin/bash
cd "$(dirname "$0")/../Mind2Web"
python download_data.py
python run.py --model ccli-sonnet --memory-mode awm \
    --memory-dir memories_awm_cmp --output-root results_awm_cmp --limit 80 "$@"
