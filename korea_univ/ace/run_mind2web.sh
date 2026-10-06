#!/bin/bash
cd "$(dirname "$0")/../Mind2Web"
python download_data.py
python run.py --model ccli-sonnet --memory-mode ace \
    --memory-dir memories_ace_cmp --output-root results_ace_cmp --limit 80 --ace-max-bullets 20 "$@"
