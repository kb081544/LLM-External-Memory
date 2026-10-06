#!/bin/bash
cd "$(dirname "$0")/../WebArena"
python pipeline_memory.py --website shopping --model ccli-sonnet --memory_mode ace \
    --memory_dir memories_ace_cmp --output_dir results_ace_cmp --end_index 80 "$@"
