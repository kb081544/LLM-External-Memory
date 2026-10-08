#!/bin/bash
cd "$(dirname "$0")/../Mind2Web"
python download_data.py
python run.py --model ccli-sonnet --memory-mode memp \
    --memory-dir memories_memp_cmp --memp-build proceduralization --memp-update adjustment \
    --memp-top-k 1 --output-root results_memp_cmp --limit 80 "$@"
