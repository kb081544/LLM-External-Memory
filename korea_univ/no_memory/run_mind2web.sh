#!/bin/bash
cd "$(dirname "$0")/../Mind2Web"
python download_data.py
python run.py --model ccli-sonnet --memory-mode none --output-root results_none_cmp --limit 80 "$@"
