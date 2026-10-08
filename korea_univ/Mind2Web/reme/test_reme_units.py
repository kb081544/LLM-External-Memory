# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");

"""Unit tests for reme/ -- fake embeddings, no real API calls.
Run directly: python reme/test_reme_units.py
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


# ------------------------------------------------------------------- extract.parse_experiences
from reme.extract import parse_experiences, format_item_for_prompt  # noqa: E402

RAW = """# Experience 1
## When to use Looking for a specific complaint inside product reviews
## Content Open the reviews tab and scan for the feature keyword before paging.
## Keywords reviews, keyword scan, pagination
## Confidence high

# Experience 2
## When to use Checking the status of a past order
## Content Go to My Account, then Orders, and match on the order date.
## Keywords orders, account
## Confidence medium
"""

items = parse_experiences(RAW)
check("parses both experiences", len(items) == 2)
check("scenario captured", items[0]["scenario"].startswith("Looking for a specific complaint"))
check("keywords split", items[0]["keywords"] == ["reviews", "keyword scan", "pagination"])
check("confidence captured", items[1]["confidence"] == "medium")

check("fenced response still parses", len(parse_experiences("```\n" + RAW + "\n```")) == 2)
check("missing optional sections tolerated", len(parse_experiences(
    "# Experience 1\n## When to use X\n## Content Y\n")) == 1)
check("block without content is dropped", parse_experiences(
    "# Experience 1\n## When to use X\n") == [])

block = format_item_for_prompt(3, items[0])
check("injection block is numbered like every other arm",
      block.startswith("# Memory Item 3\n## Title "))
check("injection block carries the content", items[0]["content"] in block)


# --------------------------------------------------------------------------- utility.credit/prune
from reme.experience import new_item, next_id  # noqa: E402
from reme.utility import credit_outcome, prune  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    events = os.path.join(tmp, "events.jsonl")
    store = {}
    for i in range(3):
        store[next_id(store)] = new_item(
            {"scenario": f"s{i}", "content": "c", "keywords": [], "confidence": ""},
            task_id=i, status="success", now=0, facet="success",
        )
    ids = list(store)

    # 5 retrievals, 1 success -> u/f = 0.2 <= 0.5 with f >= 5  => deleted
    for _ in range(5):
        store[ids[0]]["n_retrieved"] += 1
    credit_outcome([(ids[0], store[ids[0]])], judged_success=True, now=1,
                   events_path=events, task_id="t1")
    # 5 retrievals, 4 successes => kept
    for _ in range(5):
        store[ids[1]]["n_retrieved"] += 1
    for _ in range(4):
        credit_outcome([(ids[1], store[ids[1]])], judged_success=True, now=1,
                       events_path=events, task_id="t2")
    # 2 retrievals, 0 successes => below alpha, kept
    for _ in range(2):
        store[ids[2]]["n_retrieved"] += 1

    n = prune(store, now=2, events_path=events)
    check("prune deletes only the low-utility item past alpha", n == 1)
    check("low-utility item tombstoned", store[ids[0]]["deleted"] is True)
    check("deletion reason recorded", "u/f" in (store[ids[0]]["deleted_reason"] or ""))
    check("high-utility item kept", store[ids[1]]["deleted"] is False)
    check("below-alpha item kept", store[ids[2]]["deleted"] is False)
    check("credit bumps u, never f", store[ids[1]]["n_success"] == 4
          and store[ids[1]]["n_retrieved"] == 5)
    check("failure credits nothing", (
        credit_outcome([(ids[2], store[ids[2]])], judged_success=False, now=3,
                       events_path=events, task_id="t3")
        or store[ids[2]]["n_success"] == 0))
    check("events written", os.path.exists(events) and os.path.getsize(events) > 0)


# ------------------------------------------------------------------------------ retrieve (faked)
from reme import retrieve as reme_retrieve_mod  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    scen_cache = os.path.join(tmp, "scenario.jsonl")
    q_cache = os.path.join(tmp, "query.jsonl")
    store = {}
    vecs = {"e0000": [1.0, 0.0], "e0001": [0.0, 1.0], "e0002": [0.9, 0.1]}
    for i, (iid, vec) in enumerate(vecs.items()):
        store[iid] = new_item({"scenario": f"s{i}", "content": "c", "keywords": [],
                               "confidence": ""}, task_id=i, status="success", now=0,
                              facet="success")
        with open(scen_cache, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": iid, "text": f"s{i}", "embedding": vec}) + "\n")
    store["e0001"]["deleted"] = True  # must never be returned

    with patch.object(reme_retrieve_mod.efm_store, "embed_text",
                      return_value=torch.tensor([[1.0, 0.0]])):
        got = reme_retrieve_mod.retrieve(
            query="q", task_id="42", items=store, now=7, top_k=2,
            scenario_cache_path=scen_cache, query_cache_path=q_cache,
        )

    got_ids = [iid for iid, _ in got]
    check("retrieval ranks by scenario cosine", got_ids == ["e0000", "e0002"])
    check("deleted experiences are never retrieved", "e0001" not in got_ids)
    check("retrieval bumps f", store["e0000"]["n_retrieved"] == 1)
    check("query embedding cached", os.path.exists(q_cache))

    empty = reme_retrieve_mod.retrieve(
        query="q", task_id="43", items={}, now=8, top_k=2,
        scenario_cache_path=scen_cache, query_cache_path=q_cache,
    )
    check("cold bank returns nothing", empty == [])


# --------------------------------------------------------------------------- compare.find_counterpart
from reme.compare import find_counterpart  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    raw = os.path.join(tmp, "_raw.jsonl")
    rows = [
        {"task_id": "1", "website": "shopping", "status": "fail", "query": "q1"},
        {"task_id": "2", "website": "travel", "status": "success", "query": "q2"},
        {"task_id": "3", "website": "shopping", "status": "fail", "query": "q3"},
    ]
    with open(raw, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    cp = find_counterpart(raw, "shopping", "success", "4")
    check("counterpart is same website, opposite status", cp and cp["task_id"] == "3")
    check("most recent counterpart wins", cp["query"] == "q3")
    check("no counterpart for unseen website", find_counterpart(raw, "maps", "success", "4") is None)
    check("no counterpart when statuses match", find_counterpart(raw, "shopping", "fail", "4") is None)
    check("self is excluded", find_counterpart(raw, "shopping", "success", "3") and
          find_counterpart(raw, "shopping", "success", "3")["task_id"] == "1")

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
