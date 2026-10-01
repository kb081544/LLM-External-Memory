# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");

"""EFM_TASK.md 5.2-style unit tests, ported from WebArena/efm/test_efm_units.py: relation
classification x4, forgetting x2, grace, group, conflict budget/skip/cleanup -- fake embeddings,
no real API calls. Only difference from the WebArena version: task_ids use Mind2Web's
annotation_id string convention, and the single embedding mock target is `efm.store.embed_text`
(Mind2Web's store.py wraps memory.py's embed_query() behind one function, unlike WebArena's two
separate embed_query_with_gemini import sites). Run directly:
    python efm/test_efm_units.py
"""

import os
import sys
import tempfile
from unittest.mock import patch

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from efm.hparams import EFMHParams
from efm import parse as efm_parse
from efm import forget as efm_forget

passed = failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok   {name}")
    else:
        failed += 1
        print(f"  FAIL {name}")


# ---------------------------------------------------------------- parse.py
print("parse.py")
blob = [
    "# Memory Item 1\n## Title Verify search box\n"
    "## Description Use when submitting a search query.\n"
    "## Content Always check the results page loaded before continuing.",
    "# Memory Item 2\n## Title Handle stale sessions\n"
    "## Description Use when a page shows an expired-session banner.\n"
    "## Content Re-authenticate before retrying the original action.",
]
items = efm_parse.parse_memory_items(blob)
check("parses 2 items", len(items) == 2)
check("item 1 title", items[0]["title"] == "Verify search box")
check("item 1 content", "results page loaded" in items[0]["content"])
check("item 2 description", "expired-session" in items[1]["description"])

fmt = efm_parse.format_item_for_prompt(3, items[0])
check("format includes number", "Memory Item 3" in fmt)
check("format includes Description line (matches baseline's injection shape)",
      "## Description Use when submitting a search query." in fmt)


# ---------------------------------------------------------------- forget.py
print("forget.py")


def base_item(**over):
    d = {
        "title": "t", "description": "d", "content": "c", "source_label": "success",
        "created_at": 0, "last_active": 0, "n_retrieved": 0, "n_used": 0, "n_reconfirmed": 0,
        "group_id": None, "conflict_with": [], "pinned": False, "deleted": False, "deleted_reason": None,
    }
    d.update(over)
    return d


hp = EFMHParams()  # grace=20, t_stale=60, r_min=3, rho=0.34

with tempfile.TemporaryDirectory() as td:
    ev_path = os.path.join(td, "events.jsonl")

    # grace protection: brand-new item, way outside rho, must NOT be deleted yet (now-created_at < grace)
    items = {"it_0": base_item(created_at=90, last_active=90, n_retrieved=5, n_used=0)}
    n = efm_forget.forget(items, now=95, hp=hp, events_path=ev_path)
    check("grace protects new item", n == 0 and not items["it_0"]["deleted"])

    # stale deletion: last_active far in the past, past grace
    items = {"it_1": base_item(created_at=0, last_active=0, n_retrieved=1, n_used=1)}
    n = efm_forget.forget(items, now=61, hp=hp, events_path=ev_path)
    check("stale deletion fires", n == 1 and items["it_1"]["deleted_reason"] == "stale")

    # declined deletion: enough retrievals, low usage ratio, still "fresh" (not stale)
    items = {"it_2": base_item(created_at=0, last_active=25, n_retrieved=5, n_used=1)}  # 1/5=0.2 < 0.34
    n = efm_forget.forget(items, now=25, hp=hp, events_path=ev_path)
    check("declined deletion fires", n == 1 and items["it_2"]["deleted_reason"] == "declined")

    # declined rule protected by r_min: too few retrievals even with 0% usage
    items = {"it_3": base_item(created_at=0, last_active=25, n_retrieved=2, n_used=0)}
    n = efm_forget.forget(items, now=25, hp=hp, events_path=ev_path)
    check("r_min protects low-sample item", n == 0 and not items["it_3"]["deleted"])

    # pinned item is never deleted even if it qualifies for stale
    items = {"it_4": base_item(created_at=0, last_active=0, pinned=True)}
    n = efm_forget.forget(items, now=100, hp=hp, events_path=ev_path)
    check("pinned item survives", n == 0 and not items["it_4"]["deleted"])

    # conflict cleanup: deleting one side strips the survivor's conflict_with + logs conflict_resolved
    items = {
        "it_a": base_item(created_at=0, last_active=25, n_retrieved=5, n_used=4, conflict_with=["it_b"]),
        "it_b": base_item(created_at=0, last_active=25, n_retrieved=5, n_used=1, conflict_with=["it_a"]),
    }
    n = efm_forget.forget(items, now=25, hp=hp, events_path=ev_path)
    check("conflict loser deleted, winner kept", n == 1 and items["it_b"]["deleted"] and not items["it_a"]["deleted"])
    check("winner's conflict_with cleaned up", items["it_a"]["conflict_with"] == [])
    with open(ev_path) as f:
        events = [line for line in f if '"conflict_resolved"' in line]
    check("conflict_resolved event logged", len(events) == 1)

    # group cleanup: deleting a member down to 1 survivor clears that survivor's group_id
    items = {
        "it_g1": base_item(created_at=0, last_active=25, n_retrieved=5, n_used=1, group_id="grp_x"),
        "it_g2": base_item(created_at=0, last_active=25, n_retrieved=5, n_used=5, group_id="grp_x"),
    }
    efm_forget.forget(items, now=25, hp=hp, events_path=ev_path)
    check("group collapses to None when only 1 member survives", items["it_g2"]["group_id"] is None)


# ---------------------------------------------------------------- insert.py relation classification
print("insert.py (mocked embeddings)")


def fake_embed(text):
    """Deterministic fake embedding: each marker substring owns its own orthogonal dimension,
    so cosine similarity between any two markers is exactly controllable (1.0 if they share a
    dim, 0.0 if not)."""
    vec = torch.zeros(1, 8)
    markers = {
        "NOVELTAG": 0,         # duplicate test reuses this for both desc+content
        "GROUPCONTENT": 1,      # shared content marker -> high s_C for the group pair
        "GROUPDESC_A": 2,       # orthogonal per-side description -> low s_D
        "GROUPDESC_B": 3,
        "CONFLICTDESC": 4,      # shared description marker -> high s_D for the conflict pair
        "CONFLICTCONTENT_A": 5,  # orthogonal per-side content -> low s_C
        "CONFLICTCONTENT_B": 6,
    }
    for tag, dim in markers.items():
        if tag in text:
            vec[0, dim] = 1.0
    if vec.sum() == 0:
        vec[0, 7] = 1.0
    return vec


with tempfile.TemporaryDirectory() as td, \
     patch("efm.store.embed_text", side_effect=fake_embed):
    from efm import insert as efm_insert

    dcache = os.path.join(td, "desc.jsonl")
    ccache = os.path.join(td, "content.jsonl")
    events_path = os.path.join(td, "events.jsonl")
    items = {}

    class FakeClient:
        def one_step_chat(self, prompt, system_msg=None, temperature=0.0):
            # Force "contradiction" so the Conflict-candidate path is exercised deterministically.
            return '{"relation": "contradiction"}', {}

    # Novel: nothing close to it yet.
    new_ids = efm_insert.insert(
        [{"title": "Novel one", "description": "NOVELTAG desc", "content": "NOVELTAG content"}],
        task_id="m2w-ann-t1", now=0, items=items, hp=hp,
        desc_cache_path=dcache, content_cache_path=ccache, client=FakeClient(), source_label="success",
    )
    check("novel item stored", len(new_ids) == 1 and new_ids[0] in items)
    novel_id = new_ids[0]

    # Duplicate: same tag on both description and content -> matches tau_d and tau_c.
    dup_ids = efm_insert.insert(
        [{"title": "Dup of novel", "description": "NOVELTAG desc v2", "content": "NOVELTAG content v2"}],
        task_id="m2w-ann-t2", now=1, items=items, hp=hp,
        desc_cache_path=dcache, content_cache_path=ccache, client=FakeClient(), source_label="success",
    )
    check("duplicate reuses existing id, nothing new stored", dup_ids == [novel_id] and len(items) == 1)
    check("duplicate bumps n_reconfirmed", items[novel_id]["n_reconfirmed"] == 1)

    # Group: same content tag -> high s_C; orthogonal per-side description tag -> low s_D.
    g1 = efm_insert.insert(
        [{"title": "Group A", "description": "GROUPDESC_A only", "content": "GROUPCONTENT shared strategy"}],
        task_id="m2w-ann-t3", now=2, items=items, hp=hp,
        desc_cache_path=dcache, content_cache_path=ccache, client=FakeClient(), source_label="success",
    )[0]
    g2 = efm_insert.insert(
        [{"title": "Group B", "description": "GROUPDESC_B only", "content": "GROUPCONTENT shared strategy"}],
        task_id="m2w-ann-t4", now=3, items=items, hp=hp,
        desc_cache_path=dcache, content_cache_path=ccache, client=FakeClient(), source_label="success",
    )[0]
    check("group: both stored as separate items", g1 != g2)
    check("group: share a group_id", items[g1]["group_id"] is not None and items[g1]["group_id"] == items[g2]["group_id"])

    # Conflict: same description tag -> high s_D; orthogonal per-side content tag -> low s_C.
    # NLI is forced to "contradiction" by FakeClient.
    c1 = efm_insert.insert(
        [{"title": "Conflict A", "description": "CONFLICTDESC same situation", "content": "CONFLICTCONTENT_A strategy one"}],
        task_id="m2w-ann-t5", now=4, items=items, hp=hp,
        desc_cache_path=dcache, content_cache_path=ccache, client=FakeClient(), source_label="success",
    )[0]
    c2 = efm_insert.insert(
        [{"title": "Conflict B", "description": "CONFLICTDESC same situation", "content": "CONFLICTCONTENT_B strategy two"}],
        task_id="m2w-ann-t6", now=5, items=items, hp=hp,
        desc_cache_path=dcache, content_cache_path=ccache, client=FakeClient(), source_label="success",
    )[0]
    check("conflict: both stored as separate items", c1 != c2)
    check("conflict: mutually linked via conflict_with", c2 in items[c1]["conflict_with"] and c1 in items[c2]["conflict_with"])

# ---------------------------------------------------------------- retrieve.py (pure sub-steps)
print("retrieve.py (pure functions, no embeddings needed)")
from efm.retrieve import collapse_groups, pair_conflicts, allocate_budget

# group collapse: 2 of 3 share a group; the higher desc-cosine member should survive.
collapsed = collapse_groups(
    item_ids=["a", "b", "c"],
    group_of={"a": "g1", "b": "g1", "c": None},
    desc_cos_by_id={"a": 0.9, "b": 0.3},
)
check("group collapse keeps ungrouped + best-of-group", sorted(collapsed) == sorted(["a", "c"]))

# conflict pairing: a<->b are mutual conflicts and both present; c is unrelated.
pairs, singles = pair_conflicts(
    item_ids=["a", "b", "c"],
    conflict_of={"a": ["b"], "b": ["a"], "c": []},
    score_map={"a": 0.5, "b": 0.4, "c": 0.9},
)
check("pair_conflicts finds the a/b pair", len(pairs) == 1 and set(pairs[0][:2]) == {"a", "b"})
check("pair_conflicts leaves c as a single", singles == [("c", 0.9)])

# budget: pair costs 2 -- with budget=3, the higher single should NOT starve the pair's turn
# ordering (pair ranked above single here), and the leftover 1 slot goes to the single.
injected = allocate_budget(pairs=[("a", "b", 0.8)], singles=[("c", 0.5)], n_inject=3,
                            score_map={"a": 0.8, "b": 0.7, "c": 0.5})
check("budget=3: pair (2) + single (1) both fit", sorted(injected) == sorted(["a", "b", "c"]))

# budget=1 exactly on a pair's turn with a single available later -> skip the pair, take the single.
injected = allocate_budget(pairs=[("a", "b", 0.9)], singles=[("c", 0.5)], n_inject=1,
                            score_map={"a": 0.9, "b": 0.85, "c": 0.5})
check("budget=1 with a single available: pair skipped, single used", injected == ["c"])

# budget=1 exactly on a pair's turn with NO single anywhere -> inject only the higher half.
injected = allocate_budget(pairs=[("a", "b", 0.9)], singles=[], n_inject=1,
                            score_map={"a": 0.9, "b": 0.3})
check("budget=1 with no single: higher-scoring half of the pair only", injected == ["a"])


print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
