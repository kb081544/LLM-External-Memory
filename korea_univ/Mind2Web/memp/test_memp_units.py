# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");

"""Unit tests for memp/ -- fake embeddings and a fake LLM client, no real API calls.
Run directly: python memp/test_memp_units.py
"""

import json
import os
import sys
import tempfile
from unittest.mock import patch

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

passed = failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok   {name}")
    else:
        failed += 1
        print(f"  FAIL {name}")


# ------------------------------------------------------------------- procedure.parse_procedure
from memp.procedure import (  # noqa: E402
    parse_procedure, format_for_prompt, next_id, load_procedures, save_procedures,
)

RAW = """# Procedure
## When to use Finding reviews that mention a specific feature
## Steps
1. Open the product page for {item-name}
2. Click the reviews tab
3. Scan for the feature keyword
"""

proc = parse_procedure(RAW)
check("procedure parsed", proc is not None)
check("when_to_use captured", proc["when_to_use"].startswith("Finding reviews"))
check("three steps captured", len(proc["steps"]) == 3)
check("fenced response still parses", parse_procedure("```\n" + RAW + "\n```") is not None)
check("garbage returns None", parse_procedure("sorry, no procedure here") is None)
check("steps-only block returns None", parse_procedure("# Procedure\n## When to use X\n## Steps\n") is None)

full = {**proc, "trajectory": "<think>t</think>"}
script_block = format_for_prompt(1, full, "script")
traj_block = format_for_prompt(1, full, "trajectory")
both_block = format_for_prompt(1, full, "proceduralization")
check("script build injects steps only",
      "Steps:" in script_block and "Source trajectory" not in script_block)
check("trajectory build injects trajectory only",
      "Source trajectory" in traj_block and "Steps:" not in traj_block)
check("proceduralization injects both",
      "Steps:" in both_block and "Source trajectory" in both_block)
check("block is numbered like every other arm", both_block.startswith("# Memory Item 1\n## Title "))


# ------------------------------------------------------------------------------ update strategies
from memp.update import apply_update  # noqa: E402


class FakeClient:
    """Returns a revised procedure; records how many times it was called."""
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def one_step_chat(self, prompt, system_msg=None, temperature=0.0):
        self.calls += 1
        self.last_prompt = prompt
        return self.reply, None


REVISED = """# Procedure
## When to use Finding reviews that mention a specific feature
## Steps
1. Open the product page for {item-name}
2. Sort reviews by most recent first
3. Scan for the feature keyword
"""

with tempfile.TemporaryDirectory() as tmp:
    events = os.path.join(tmp, "events.jsonl")

    # vanilla: appends on failure too
    procs = {}
    client = FakeClient(REVISED)
    pid, adjusted = apply_update("vanilla", procs, proc, judged_success=False, injected=[],
                                 query="q", trajectory="traj", task_id="1", now=0,
                                 client=client, events_path=events)
    check("vanilla appends a failed task's procedure", pid is not None and len(procs) == 1)
    check("vanilla makes no LLM call", client.calls == 0)
    check("source status recorded", procs[pid]["source_status"] == "fail")

    # validation: skips on failure, appends on success
    procs = {}
    pid, _ = apply_update("validation", procs, proc, judged_success=False, injected=[],
                          query="q", trajectory="traj", task_id="2", now=0,
                          client=client, events_path=events)
    check("validation skips a failed task", pid is None and procs == {})
    pid, _ = apply_update("validation", procs, proc, judged_success=True, injected=[],
                          query="q", trajectory="traj", task_id="3", now=0,
                          client=client, events_path=events)
    check("validation appends a successful task", pid is not None and len(procs) == 1)

    # adjustment: success appends, failure rewrites the injected procedure in place
    procs = {}
    first, _ = apply_update("adjustment", procs, proc, judged_success=True, injected=[],
                            query="q", trajectory="traj", task_id="4", now=0,
                            client=client, events_path=events)
    before_steps = list(procs[first]["steps"])
    client = FakeClient(REVISED)
    pid, adjusted = apply_update(
        "adjustment", procs, proc, judged_success=False,
        injected=[(first, procs[first])], query="q", trajectory="traj", task_id="5", now=1,
        client=client, events_path=events,
    )
    check("adjustment appends nothing on failure", pid is None)
    check("adjustment rewrites in place", adjusted is True and len(procs) == 1)
    check("steps actually changed", procs[first]["steps"] != before_steps)
    check("revision counted", procs[first]["n_revisions"] == 1)
    check("reviser task recorded", procs[first]["revised_by"] == ["5"])
    check("adjustment costs exactly one LLM call", client.calls == 1)
    check("adjust prompt shows the old procedure and the failed run",
          "EXISTING PROCEDURE" in client.last_prompt and "FAILED ATTEMPT" in client.last_prompt)

    # adjustment with a cold bank has nothing to blame -> behaves like validation
    procs = {}
    client = FakeClient(REVISED)
    pid, adjusted = apply_update("adjustment", procs, proc, judged_success=False, injected=[],
                                 query="q", trajectory="traj", task_id="6", now=2,
                                 client=client, events_path=events)
    check("adjustment with no injection makes no edit and no call",
          pid is None and adjusted is False and client.calls == 0)

    # unparseable revision leaves the stored procedure untouched
    procs = {}
    first, _ = apply_update("adjustment", procs, proc, judged_success=True, injected=[],
                            query="q", trajectory="traj", task_id="7", now=3,
                            client=FakeClient(REVISED), events_path=events)
    kept = list(procs[first]["steps"])
    pid, adjusted = apply_update(
        "adjustment", procs, proc, judged_success=False, injected=[(first, procs[first])],
        query="q", trajectory="traj", task_id="8", now=4,
        client=FakeClient("sorry, nothing here"), events_path=events,
    )
    check("unparseable revision is discarded", adjusted is False and procs[first]["steps"] == kept)

    raised = False
    try:
        apply_update("nope", {}, proc, judged_success=True, injected=[], query="q",
                     trajectory="t", task_id="9", now=0, client=client, events_path=events)
    except ValueError:
        raised = True
    check("unknown update mode raises ValueError", raised)


# ---------------------------------------------------------------------------- retrieve (faked)
from memp import retrieve as memp_retrieve_mod  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    key_cache = os.path.join(tmp, "key.jsonl")
    q_cache = os.path.join(tmp, "query.jsonl")
    procs = {}
    for pid, vec in {"p0000": [1.0, 0.0], "p0001": [0.0, 1.0], "p0002": [0.8, 0.2]}.items():
        procs[pid] = {"when_to_use": pid, "steps": ["1. x"], "trajectory": "",
                      "deleted": False, "n_retrieved": 0}
        with open(key_cache, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": pid, "text": pid, "embedding": vec}) + "\n")
    procs["p0002"]["deleted"] = True

    with patch.object(memp_retrieve_mod.efm_store, "embed_text",
                      return_value=torch.tensor([[1.0, 0.0]])):
        got = memp_retrieve_mod.retrieve(
            query="q", task_id="10", procs=procs, now=0, top_k=1,
            key_cache_path=key_cache, query_cache_path=q_cache,
        )
    check("top-1 retrieval by query cosine", [p for p, _ in got] == ["p0000"])
    check("deleted procedures are never retrieved", "p0002" not in [p for p, _ in got])
    check("retrieval counted", procs["p0000"]["n_retrieved"] == 1)


# ----------------------------------------------------------------------------- store round-trip
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "procedures.json")
    save_procedures(path, {"p0000": {"when_to_use": "x", "steps": ["1. y"]}})
    back = load_procedures(path)
    check("store round-trips", back["p0000"]["steps"] == ["1. y"])
    check("next_id avoids collisions", next_id(back) == "p0001")
    check("missing store reads as empty", load_procedures(os.path.join(tmp, "nope.json")) == {})

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
