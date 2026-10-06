# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");

"""ACE's "Curator" step: merges the Reflector's freshly-extracted candidate bullets into the
persistent playbook. The real ACE shows the Reflector the current playbook so it can vote on
existing bullet IDs directly; here the Reflector call (induce_memory.py) is kept stateless like
reasoningbank/awm/synapse (see prompts/memory_instruction.py's docstring), so dedup-vs-existing
happens here instead, via the same embedding-cosine approach WebArena/efm/insert.py uses for its
own Duplicate classification -- reused directly, not reinvented."""

import re

from memory_management import embed_query_with_gemini
from efm import store as efm_store

_TAU = 0.85  # same default as EFM_TASK.md's TAU_D/TAU_C -- "clearly the same strategy"
_BULLET_RE = re.compile(r"\[(?P<section>\w+)\]\s*(?P<content>.+)", re.DOTALL)


def parse_candidate_bullets(raw_text: str) -> list[dict]:
    # Defense-in-depth: induce_memory.py already strips a whole-response code fence, but strip
    # any stray ``` lines per-block too (e.g. a model fencing just the first/last bullet).
    raw_text = re.sub(r"^```[a-zA-Z]*$", "", raw_text, flags=re.MULTILINE)
    blocks = [b.strip() for b in raw_text.split("\n\n") if b.strip()]
    out = []
    for block in blocks:
        block = block.strip("` \n")
        m = _BULLET_RE.match(block)
        if not m:
            continue
        section = m.group("section").strip().lower()
        content = m.group("content").strip().strip("`").strip()
        if content:
            out.append({"section": section, "content": content})
    return out


def curate(candidates: list[dict], playbook: dict, task_succeeded: bool,
           cache_path: str) -> int:
    """Mutates `playbook` in place. Returns the number of brand-new bullets added (as opposed
    to merged into an existing one via a helpful/harmful vote bump)."""
    from ace.playbook import next_bullet_id

    cache_ids, _texts, cache_emb = efm_store.load_cache(cache_path)
    added = 0
    for cand in candidates:
        # Plain content embedding, same as EFM's own insert.py uses for its description/content
        # vectors -- embed_query_for_scoring's instruction-wrap is for query-vs-document
        # retrieval scoring, not content-vs-content similarity, so it doesn't apply here.
        c_vec = embed_query_with_gemini(cand["content"], dimensionality=3072)
        best_id, best_sim = None, -1.0
        if len(cache_ids):
            sims = efm_store.cosine_to_cache(c_vec, cache_emb).tolist()
            for bid, sim in zip(cache_ids, sims):
                if bid in playbook and sim > best_sim:
                    best_id, best_sim = bid, sim

        if best_id is not None and best_sim >= _TAU:
            # Clearly the same strategy already in the playbook -- vote, don't duplicate.
            if task_succeeded:
                playbook[best_id]["helpful"] += 1
            else:
                playbook[best_id]["harmful"] += 1
            continue

        new_id = next_bullet_id(playbook)
        playbook[new_id] = {
            "section": cand["section"],
            "content": cand["content"],
            "helpful": 1 if task_succeeded else 0,
            "harmful": 0 if task_succeeded else 1,
        }
        efm_store.append_cache(c_vec, cache_path, new_id, cand["content"])
        added += 1

    return added
