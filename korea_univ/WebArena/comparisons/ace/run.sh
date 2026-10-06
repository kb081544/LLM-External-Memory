#!/bin/bash
# ace arm: Agentic Context Engineering (arXiv:2510.04618) reimplementation -- evolving
# playbook of helpful/harmful-voted bullets, injected whole (size-capped) every task.
cd "$(dirname "$0")/../.."
python pipeline_memory.py --website shopping --model ccli-sonnet --memory_mode ace \
    --memory_dir memories_ace_cmp --output_dir results_ace_cmp --end_index 80 "$@"
