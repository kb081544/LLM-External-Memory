# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");

"""ReMe (Remember Me, Refine Me, Findings of ACL 2026) experience store.

Reimplementation, not an import of agentscope-ai/ReMe: the released code targets BFCL-V3 /
AppWorld tool-calling episodes and ships its own agent loop + vector store, neither of which
maps onto WebArena's browser episode loop. Only the memory algorithm is reproduced here, the
same way WebArena/ace/ reproduces ACE (see prompts/memory_instruction.py).

Experience schema follows the paper's E = <omega, e, kappa, c, tau>:
  scenario  (omega) -- "when to use this", the field ReMe indexes retrieval on (paper 4.3:
                       usage-scenario indexing beat raw query / keyword indexing)
  content   (e)     -- the reusable insight itself
  keywords  (kappa) -- extracted keywords, kept for inspection/analysis
  confidence(c)     -- model-stated confidence, kept for inspection
  tools     (tau)   -- action types touched by the source trajectory

Utility bookkeeping (paper 3.4, Eq. 1): every retrieval bumps `n_retrieved` (f) and a
retrieval that lands in a successful task bumps `n_success` (u); deletion is f >= alpha and
u/f <= beta. `n_used` is NOT part of ReMe -- it is recorded only for cross-arm analysis
against efm, and never feeds the deletion rule.
"""

import json
import os


def load_items(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_items(path: str, items: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)


def next_id(items: dict) -> str:
    n = len(items)
    while f"e{n:04d}" in items:
        n += 1
    return f"e{n:04d}"


def new_item(parsed: dict, task_id, status: str, now: int, facet: str) -> dict:
    """`facet` is which of ReMe's three extraction analyses produced this item:
    success / failure / comparative (paper 3.2)."""
    return {
        "scenario": parsed["scenario"],
        "content": parsed["content"],
        "keywords": parsed.get("keywords", []),
        "confidence": parsed.get("confidence", ""),
        "tools": parsed.get("tools", []),
        "facet": facet,
        "source_task": str(task_id),
        "source_status": status,
        "created": now,
        "n_retrieved": 0,
        "n_success": 0,
        "n_used": 0,
        "deleted": False,
        "deleted_reason": None,
    }


def log_event(events_path: str, event: dict) -> None:
    os.makedirs(os.path.dirname(events_path) or ".", exist_ok=True)
    with open(events_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
