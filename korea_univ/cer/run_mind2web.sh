#!/bin/bash
cd "$(dirname "$0")/../Mind2Web"
python download_data.py
python run.py --model ccli-sonnet --memory-mode cer \
    --memory-dir memories_cer_cmp --cer-max-skills 5 \
    --output-root results_cer_cmp --limit 80 "$@"
