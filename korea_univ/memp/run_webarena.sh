#!/bin/bash
cd "$(dirname "$0")/../WebArena"
python pipeline_memory.py --website shopping --model ccli-sonnet --memory_mode memp \
    --memory_dir memories_memp_cmp --output_dir results_memp_cmp --end_index 80 \
    --memp_top_k 1 --memp_build proceduralization --memp_update adjustment "$@"
