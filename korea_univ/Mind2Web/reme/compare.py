# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");

"""ReMe's third extraction analysis: **comparative** (paper 3.2) -- "comparative analysis by
jointly examining successful and failed trajectories".

The paper gets its success/fail pair by sampling the same training task N times. Our runner
executes each Mind2Web task exactly once (one trajectory per task, shared by every arm, so the
arms stay comparable), so the pair is formed across tasks instead: the task just finished is
contrasted with the most recent earlier task on the SAME WEBSITE that had the opposite outcome.
WebArena's port keys this on `intent_template_id`; Mind2Web rows carry no template id, and
`website` is the closest available "same kind of task" marker.

If no such counterpart exists yet, no comparative call is made (one extra LLM call only when a
genuine pair is available, so the cost column stays honest).
"""

import json
import os

from memory_instruction import REME_COMPARATIVE_SI


def find_counterpart(raw_path: str, website, status: str, task_id) -> dict | None:
    """Scan the arm's _raw.jsonl for the most recent earlier entry on the same website with
    the OPPOSITE status. Returns that raw entry, or None."""
    if not os.path.exists(raw_path):
        return None
    want = "fail" if status == "success" else "success"
    best = None
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(entry.get("task_id")) == str(task_id):
                continue
            if entry.get("website") != website:
                continue
            if entry.get("status") != want:
                continue
            best = entry  # later lines win -> most recent counterpart
    return best


def format_pair(success_entry: dict, fail_entry: dict) -> str:
    def _traj(entry):
        # Mind2Web entries store the already-formatted trajectory text (run.py builds it with
        # memory.format_trajectory_for_induction), unlike WebArena's think_list/action_list.
        return entry.get("trajectory", "")

    return (
        f"## SUCCESSFUL RUN\n**Query:** {success_entry['query']}\n\n"
        f"**Trajectory:**\n{_traj(success_entry)}\n\n"
        f"## FAILED RUN\n**Query:** {fail_entry['query']}\n\n"
        f"**Trajectory:**\n{_traj(fail_entry)}"
    )


def comparative_extract(client, cur_entry: dict, counterpart: dict) -> str:
    """One LLM call contrasting the pair. Returns raw text for reme/extract.py to parse."""
    if cur_entry["status"] == "success":
        success_entry, fail_entry = cur_entry, counterpart
    else:
        success_entry, fail_entry = counterpart, cur_entry
    text, _ = client.one_step_chat(
        format_pair(success_entry, fail_entry),
        system_msg=REME_COMPARATIVE_SI,
        temperature=0.7,
    )
    return text or ""
