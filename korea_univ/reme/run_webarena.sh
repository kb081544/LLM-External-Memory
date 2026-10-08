#!/bin/bash
cd "$(dirname "$0")/../WebArena"
python pipeline_memory.py --website shopping --model ccli-sonnet --memory_mode reme \
    --memory_dir memories_reme_cmp --output_dir results_reme_cmp --end_index 80 \
    --reme_top_k 3 "$@"
