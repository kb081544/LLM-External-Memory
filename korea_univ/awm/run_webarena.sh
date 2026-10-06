#!/bin/bash
cd "$(dirname "$0")/../WebArena"
python pipeline_memory.py --website shopping --model ccli-sonnet --memory_mode awm \
    --memory_dir memories_awm_cmp --output_dir results_awm_cmp --end_index 80 "$@"
