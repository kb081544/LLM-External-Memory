# Copyright 2026 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Usage-based utility scoring and pruning for the reasoningbank_pruned memory mode.

Heuristic: a memory bank entry (one task_id's induced memory_items) accrues a
retrieval_count each time it's selected by select_memory() for some other task's query
(memory_management.log_retrieval_event), and a success_count each time the task that
retrieved it went on to succeed. Utility = success_count / retrieval_count, protected by a
minimum-sample floor so a rarely-retrieved entry isn't pruned on noise.

Credit assignment is approximate by design: a task retrieves top-N entries jointly, and
success/failure is credited to all of them equally. This is a starting heuristic, not a
causal claim about which entry actually helped.

Only used by pipeline_memory.py when --memory_mode reasoningbank_pruned; no other mode
imports this module.
"""

import json
import os


def _load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r") as f:
        return json.load(f)


def _load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def update_usage_stats(stats_path: str, usage_log_path: str, outcome_task_id: str, success: bool) -> None:
    """Find the retrieval event(s) logged for `outcome_task_id`'s query in usage_log_path,
    and increment retrieval_count / success_count for each donor task_id it retrieved. If a
    task was retried (idle-timeout retry in pipeline_memory.py), only the most recent
    logged retrieval for that task_id is counted, to avoid double-crediting."""
    stats = _load_json(stats_path, {})

    donor_ids = None
    for event in _load_jsonl(usage_log_path):
        if event["query_task_id"] == outcome_task_id:
            donor_ids = event["retrieved_task_ids"]

    if not donor_ids:
        return

    for donor_id in donor_ids:
        entry = stats.setdefault(donor_id, {"retrieval_count": 0, "success_count": 0})
        entry["retrieval_count"] += 1
        if success:
            entry["success_count"] += 1

    os.makedirs(os.path.dirname(stats_path) or ".", exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)


def compute_utility(stats: dict) -> dict:
    """task_id -> success_count / retrieval_count, for entries with at least one retrieval."""
    return {
        task_id: entry["success_count"] / entry["retrieval_count"]
        for task_id, entry in stats.items()
        if entry["retrieval_count"] > 0
    }


def prune_memory_bank(memory_bank_path: str, stats: dict, min_samples: int = 5, threshold: float = 0.3) -> int:
    """Rewrite memory_bank_path, dropping entries whose task_id has enough samples
    (retrieval_count >= min_samples) but a low utility score (< threshold). Removed entries
    are appended -- not silently discarded -- to <bank>_pruned.jsonl with their final stats,
    for auditability. Returns the number of entries pruned."""
    entries = _load_jsonl(memory_bank_path)
    if not entries:
        return 0

    kept, pruned = [], []
    for entry in entries:
        entry_stats = stats.get(entry.get("task_id"), {"retrieval_count": 0, "success_count": 0})
        retrieval_count = entry_stats["retrieval_count"]
        utility = entry_stats["success_count"] / retrieval_count if retrieval_count else None

        if retrieval_count >= min_samples and utility is not None and utility < threshold:
            pruned.append({**entry, "final_stats": entry_stats, "final_utility": utility})
        else:
            kept.append(entry)

    if not pruned:
        return 0

    with open(memory_bank_path, "w") as f:
        for entry in kept:
            f.write(json.dumps(entry) + "\n")

    pruned_log_path = memory_bank_path.rsplit(".jsonl", 1)[0] + "_pruned.jsonl"
    with open(pruned_log_path, "a") as f:
        for entry in pruned:
            f.write(json.dumps(entry) + "\n")

    return len(pruned)
