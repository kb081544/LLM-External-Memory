#!/bin/bash
cd "$(dirname "$0")/../WebArena"
python pipeline_memory.py --website shopping --model ccli-sonnet --memory_mode no_memory \
    --output_dir results_no_memory_cmp --end_index 80 "$@"
