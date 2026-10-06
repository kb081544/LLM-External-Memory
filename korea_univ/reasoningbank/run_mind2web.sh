#!/bin/bash
cd "$(dirname "$0")/../Mind2Web"
python download_data.py
python run.py --model ccli-sonnet --memory-mode reasoningbank \
    --memory-dir memories_reasoningbank_cmp --output-root results_reasoningbank_cmp --limit 80 "$@"
