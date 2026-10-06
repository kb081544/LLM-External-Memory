#!/bin/bash
# reasoningbank arm: baseline -- ReasoningBank paper's method (distills from success+failure,
# injects the nearest past experience's items verbatim, no management/forgetting).
cd "$(dirname "$0")/../.."
python pipeline_memory.py --website shopping --model ccli-sonnet --memory_mode reasoningbank \
    --memory_dir memories_reasoningbank_cmp --output_dir results_reasoningbank_cmp --end_index 80 "$@"
