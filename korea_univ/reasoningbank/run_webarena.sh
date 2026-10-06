#!/bin/bash
cd "$(dirname "$0")/../WebArena"
python pipeline_memory.py --website shopping --model ccli-sonnet --memory_mode reasoningbank \
    --memory_dir memories_reasoningbank_cmp --output_dir results_reasoningbank_cmp --end_index 80 "$@"
