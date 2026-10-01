# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");

"""EFM_TASK.md 3.5 -- insertion-time relation classification (Duplicate/Group/Conflict/Novel)
for freshly-extracted items. The extraction call itself (induce_memory.py, SUCCESSFUL_SI/
FAILED_SI) is never touched; this only decides what happens to its *output* before storage."""

import json
import uuid

import torch

from memory_management import embed_query_with_gemini
from efm import store

NLI_PROMPT = """Two strategies apply to the same situation.
Situation A: {desc_old}
Strategy A: {content_old}
Situation B: {desc_new}
Strategy B: {content_new}
Do the two strategies contradict each other (following one means not following the other),
or are they complementary (both can be followed or either works)?
Answer with JSON only: {{"relation": "contradiction"}} or {{"relation": "complementary"}}"""


def _nli_relation(client, old_item: dict, new_item: dict) -> str:
    prompt = NLI_PROMPT.format(
        desc_old=old_item["description"], content_old=old_item["content"],
        desc_new=new_item["description"], content_new=new_item["content"],
    )
    text, _raw = client.one_step_chat(prompt, system_msg=None, temperature=0.0)
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        relation = json.loads(text[start:end]).get("relation", "complementary")
    except Exception:
        relation = "complementary"  # fail open: treat unparseable NLI output as Novel, not a false conflict
    return relation if relation in ("contradiction", "complementary") else "complementary"


def _top_m(pool: dict[str, float], m: int) -> list[str]:
    return [iid for iid, _ in sorted(pool.items(), key=lambda x: -x[1])[:m]]


def insert(new_items_raw: list[dict], task_id: str, now: int, items: dict, hp,
           desc_cache_path: str, content_cache_path: str, client, source_label: str) -> list[str]:
    """Returns the item_ids to record on this task's experience (existing ids for Duplicates,
    new ids otherwise)."""
    desc_ids, _dt, desc_emb = store.load_cache(desc_cache_path)
    content_ids, _ct, content_emb = store.load_cache(content_cache_path)
    active = {iid for iid, it in items.items() if not it["deleted"]}

    batch_desc_vec: dict[str, list] = {}
    batch_content_vec: dict[str, list] = {}
    experience_item_ids: list[str] = []

    for idx, new in enumerate(new_items_raw):
        d_vec = embed_query_with_gemini(new["description"], dimensionality=3072)
        c_vec = embed_query_with_gemini(new["content"], dimensionality=3072)

        s_c_existing = {
            iid: float(store.cosine_to_cache(c_vec, content_emb[[content_ids.index(iid)]])[0])
            for iid in active if iid in content_ids
        } if len(content_ids) else {}
        s_d_existing = {
            iid: float(store.cosine_to_cache(d_vec, desc_emb[[desc_ids.index(iid)]])[0])
            for iid in active if iid in desc_ids
        } if len(desc_ids) else {}

        candidate_ids = set(_top_m(s_c_existing, hp.m_cand)) | set(_top_m(s_d_existing, hp.m_cand))
        candidate_ids |= set(batch_desc_vec.keys())  # all earlier items in this same batch

        s_d, s_c = {}, {}
        for cid in candidate_ids:
            if cid in batch_desc_vec:
                s_d[cid] = float((torch.tensor(batch_desc_vec[cid]) * d_vec.squeeze(0)).sum()
                                  / (torch.tensor(batch_desc_vec[cid]).norm() * d_vec.norm()))
                s_c[cid] = float((torch.tensor(batch_content_vec[cid]) * c_vec.squeeze(0)).sum()
                                  / (torch.tensor(batch_content_vec[cid]).norm() * c_vec.norm()))
            else:
                s_d[cid] = s_d_existing.get(cid, -1.0)
                s_c[cid] = s_c_existing.get(cid, -1.0)

        dup = [c for c in candidate_ids if s_d[c] >= hp.tau_d and s_c[c] >= hp.tau_c]
        if dup:
            best = max(dup, key=lambda c: s_c[c])
            items[best]["n_reconfirmed"] += 1
            items[best]["last_active"] = now
            experience_item_ids.append(best)
            continue

        new_id = store.next_id(items, "it")
        record = {
            "title": new["title"], "description": new["description"], "content": new["content"],
            "source_label": source_label,
            "created_at": now, "last_active": now,
            "n_retrieved": 0, "n_used": 0, "n_reconfirmed": 0,
            "group_id": None, "conflict_with": [], "pinned": False,
            "deleted": False, "deleted_reason": None,
        }

        group_cands = [c for c in candidate_ids if s_d[c] < hp.tau_d and s_c[c] >= hp.tau_c]
        if group_cands:
            gid = next((items[c]["group_id"] for c in group_cands if items.get(c, {}).get("group_id")), None)
            gid = gid or f"grp_{uuid.uuid4().hex[:8]}"
            record["group_id"] = gid
            for c in group_cands:
                if c in items:
                    items[c]["group_id"] = gid

        conflict_cands = [c for c in candidate_ids if s_d[c] >= hp.tau_d and s_c[c] < hp.tau_c]
        for c in conflict_cands:
            if c not in items:
                continue  # batch-mate not yet persisted into `items`; nothing to link against
            if _nli_relation(client, items[c], record) == "contradiction":
                record["conflict_with"].append(c)
                items[c]["conflict_with"].append(new_id)

        items[new_id] = record
        store.append_cache(d_vec, desc_cache_path, new_id, new["description"])
        store.append_cache(c_vec, content_cache_path, new_id, new["content"])
        batch_desc_vec[new_id] = d_vec.squeeze(0).tolist()
        batch_content_vec[new_id] = c_vec.squeeze(0).tolist()
        experience_item_ids.append(new_id)

    return experience_item_ids
