#!/bin/bash
cd "$(dirname "$0")/../WebArena"
python pipeline_memory.py --website shopping --model ccli-sonnet --memory_mode cer \
    --memory_dir memories_cer_cmp --output_dir results_cer_cmp --end_index 80 \
    --cer_max_dynamics 5 --cer_max_skills 5 "$@"
