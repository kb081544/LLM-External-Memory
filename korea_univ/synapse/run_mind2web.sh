#!/bin/bash
cd "$(dirname "$0")/../Mind2Web"
python download_data.py
python run.py --model ccli-sonnet --memory-mode synapse \
    --memory-dir memories_synapse_cmp --output-root results_synapse_cmp --limit 80 "$@"
