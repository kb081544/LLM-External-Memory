# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");

"""Memp retrieval (paper 3.1 / 4.2): m_retrieved = argmax S(t_new, t_i) with S = cosine over
the task-query embedding -- the paper's "Key=Query" condition, which its own ablation found
competitive with the keyword/fact variants.

The Facts and AveFact keys are NOT ported: both need a per-procedure extracted fact list and
an extra keyword-extraction call per retrieval, which would change the cost column this
comparison is built around. Noted in comparisons/README.md.

Embedding model and cache conventions come from efm/store.py (which wraps memory.py), so
all arms share one embedding model by construction.
"""

from efm import store as efm_store


def retrieve(query: str, task_id, procs: dict, now: int, top_k: int,
             key_cache_path: str, query_cache_path: str) -> list[tuple[str, dict]]:
    """Returns [(proc_id, proc)] for the top_k procedures, highest cosine first. Mutates
    `procs` in place (n_retrieved / last_retrieved) for post-hoc analysis."""
    live = {pid: p for pid, p in procs.items() if not p.get("deleted")}
    if not live:
        return []

    q_vec = efm_store.embed_text(query)
    efm_store.append_cache(q_vec, query_cache_path, f"q_{task_id}", query)

    cache_ids, _texts, cache_emb = efm_store.load_cache(key_cache_path)
    if cache_emb.numel() == 0:
        return []

    sims = efm_store.cosine_to_cache(q_vec, cache_emb).tolist()
    best: dict[str, float] = {}
    for pid, sim in zip(cache_ids, sims):
        if pid in live and sim > best.get(pid, -2.0):
            best[pid] = sim

    ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    out = []
    for pid, _sim in ranked:
        proc = live[pid]
        proc["n_retrieved"] = proc.get("n_retrieved", 0) + 1
        proc["last_retrieved"] = now
        out.append((pid, proc))
    return out
