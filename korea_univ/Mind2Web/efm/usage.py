# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");

"""EFM_TASK.md 3.4 -- usage declaration recording. The agent is never given new instructions
for this (agent.py already asks it to "discuss if you want to use each memory item or not" --
see CODE_MAP.md); this module only interprets that existing free-text response.

Design note (see NOTES.md): 3.4.1 offers two options, rule-based parsing of "Memory Item N"
mentions or one LLM classification call as a fallback. A robust rule-based mention-context
parser (distinguishing "I'll use Item 2" from "Item 2 doesn't apply here") is fragile to get
right; this implementation uses the LLM-classification path as the primary method instead of
only a last-resort fallback. The failure path (call errors or output doesn't parse) still
defaults every item to "used", exactly as 3.4.1 specifies for the truly-unparseable case."""

import json
import os


def _log_event(events_path: str, event: dict) -> None:
    os.makedirs(os.path.dirname(events_path) or ".", exist_ok=True)
    with open(events_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def record_usage(injected: list[tuple[str, dict]], think_text: str, now: int, client,
                  events_path: str, task_id, judged_success: bool) -> None:
    """injected: [(item_id, item)] in the order/numbering shown to the agent (1-indexed
    display). Mutates n_used/last_active on used items in place."""
    listing = "\n".join(f"{n}. {item['title']}" for n, (_iid, item) in enumerate(injected, start=1))
    prompt = (
        "Below is an agent's reasoning trace, followed by a numbered list of memory items that "
        "were available to it. For each numbered item, decide whether the agent's reasoning "
        "indicates it was used (followed/applied) or declined (considered but not applied, or "
        "never mentioned).\n\n"
        f"Reasoning trace:\n{think_text}\n\n"
        f"Memory items:\n{listing}\n\n"
        'Respond with JSON only, one entry per number, e.g. {"1": "used", "2": "declined"}.'
    )
    decisions = {}
    if injected:
        try:
            text, _raw = client.one_step_chat(prompt, system_msg=None, temperature=0.0)
            start, end = text.index("{"), text.rindex("}") + 1
            decisions = json.loads(text[start:end])
        except Exception:
            decisions = {}  # unparseable -> every item below defaults to "used" (3.4.1)

    used_ids, declined_ids = set(), set()
    for n, (iid, item) in enumerate(injected, start=1):
        decision = str(decisions.get(str(n), "used")).lower()
        if decision == "declined":
            declined_ids.add(iid)
        else:
            item["n_used"] += 1
            item["last_active"] = now
            used_ids.add(iid)

    injected_ids = {iid for iid, _item in injected}
    logged_pairs = set()
    for iid, item in injected:
        for partner in item.get("conflict_with", []):
            if partner not in injected_ids:
                continue
            key = tuple(sorted((iid, partner)))
            if key in logged_pairs:
                continue
            logged_pairs.add(key)
            _log_event(events_path, {
                "type": "conflict_exposure",
                "task_id": task_id,
                "pair": list(key),
                "used": [x for x in key if x in used_ids],
                "declined": [x for x in key if x in declined_ids],
                "judge": "success" if judged_success else "fail",
            })
