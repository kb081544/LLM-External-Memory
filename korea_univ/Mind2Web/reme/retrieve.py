# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");

"""ReMe retrieval (paper 3.3 + 4.3): embed the current task query, score it against each
experience's **usage-scenario** embedding, return the top-k.

Indexing on the usage scenario rather than on the raw task query is ReMe's own ablation
result ("the usage scenario indexing strategy ... proves to be the most effective", 4.3) and
is the main retrieval-side difference from reasoningbank, which embeds the *query* of the
source task. Embedding model/cache conventions are reused from efm/store.py, which wraps memory.py --
so every arm shares one embedding model by construction. Mind2Web's baseline scores with a
plain query embedding (no instruction-wrap asymmetry), so embed_text() is used for both
caching and scoring, exactly as the efm port does.

The optional reranker / rewriter of paper 3.3 is NOT ported: the paper itself marks both as
optional, and each would add an LLM call per task, which would confound the cost column we
are comparing arms on. Noted in comparisons/README.md.
"""

from efm import store as efm_store
from reme import experience as reme_store


def retrieve(query: str, task_id, items: dict, now: int, top_k: int,
             scenario_cache_path: str, query_cache_path: str) -> list[tuple[str, dict]]:
    """Returns [(item_id, item)] for the top_k live experiences, highest cosine first.
    Mutates `items` in place: every returned item's n_retrieved (ReMe's f) is bumped."""
    live = {iid: it for iid, it in items.items() if not it.get("deleted")}
    if not live:
        # Still cache the query embedding so the cold-start case leaves the same trail the
        # other arms leave (baseline's screening() has the same side effect).
        return []

    q_vec = efm_store.embed_text(query)
    efm_store.append_cache(q_vec, query_cache_path, f"q_{task_id}", query)

    cache_ids, _texts, cache_emb = efm_store.load_cache(scenario_cache_path)
    if cache_emb.numel() == 0:
        return []

    sims = efm_store.cosine_to_cache(q_vec, cache_emb).tolist()
    best: dict[str, float] = {}
    for iid, sim in zip(cache_ids, sims):
        if iid in live and sim > best.get(iid, -2.0):
            best[iid] = sim

    ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    out = []
    for iid, _sim in ranked:
        item = live[iid]
        item["n_retrieved"] += 1
        item["last_retrieved"] = now
        out.append((iid, item))
    return out
