#!/bin/bash
# synapse arm: raw successful trajectory stored verbatim as "memory" (no distillation).
cd "$(dirname "$0")/../.."
python pipeline_memory.py --website shopping --model ccli-sonnet --memory_mode synapse \
    --memory_dir memories_synapse_cmp --output_dir results_synapse_cmp --end_index 80 "$@"
