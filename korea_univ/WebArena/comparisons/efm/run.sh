#!/bin/bash
# efm arm: our research -- Edit-Free Memory Lifecycle (insertion-time relation classification,
# retrieval-time group collapse + conflict co-injection, usage-evidence-based forgetting).
cd "$(dirname "$0")/../.."
python pipeline_memory.py --website shopping --model ccli-sonnet --memory_mode efm \
    --memory_dir memories_efm_cmp --output_dir results_efm_cmp --end_index 80 --efm_n_inject 3 "$@"
