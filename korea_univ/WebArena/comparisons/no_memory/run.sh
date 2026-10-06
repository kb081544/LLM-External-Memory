#!/bin/bash
# no_memory arm: pure agent, no memory at all (lower-bound baseline).
cd "$(dirname "$0")/../.."
python pipeline_memory.py --website shopping --model ccli-sonnet --memory_mode no_memory \
    --output_dir results_no_memory_cmp --end_index 80 "$@"
