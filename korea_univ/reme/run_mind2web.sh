#!/bin/bash
cd "$(dirname "$0")/../Mind2Web"
python download_data.py
python run.py --model ccli-sonnet --memory-mode reme \
    --memory-dir memories_reme_cmp --output-root results_reme_cmp --limit 80 --reme-top-k 3 "$@"
