# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");

"""EFM_TASK.md 3.3 -- retrieval-time candidate gathering, recency scoring, group collapse,
conflict co-injection, and budget-limited injection. Mutates n_retrieved on injected items;
caller is responsible for persisting `items` afterward."""

from efm import store


def collapse_groups(item_ids: list[str], group_of: dict, desc_cos_by_id: dict) -> list[str]:
    """Pure step 4: ungrouped ids pass through; grouped ids collapse to the single member with
    highest query-vs-description cosine. Extracted for direct unit testing (no embeddings/I-O)."""
    by_group: dict[str, list[str]] = {}
    out: list[str] = []
    for iid in item_ids:
        gid = group_of.get(iid)
        if gid:
            by_group.setdefault(gid, []).append(iid)
        else:
            out.append(iid)
    for member_ids in by_group.values():
        out.append(max(member_ids, key=lambda i: desc_cos_by_id.get(i, float("-inf"))))
    return out


def pair_conflicts(item_ids: list[str], conflict_of: dict, score_map: dict):
    """Pure step 5: co-pair items that are each other's conflict partner and both survived
    collapse; everything else is a single. Never arbitrates by recency (EFM_TASK.md 3.3.5)."""
    item_set = set(item_ids)
    paired: set[str] = set()
    pairs: list[tuple[str, str, float]] = []
    singles: list[tuple[str, float]] = []
    for iid in item_ids:
        if iid in paired:
            continue
        partner = next((c for c in conflict_of.get(iid, []) if c in item_set), None)
        if partner is not None:
            paired.add(iid)
            paired.add(partner)
            pairs.append((iid, partner, max(score_map[iid], score_map.get(partner, score_map[iid]))))
        else:
            singles.append((iid, score_map[iid]))
    return pairs, singles


def allocate_budget(pairs, singles, n_inject: int, score_map: dict) -> list[str]:
    """Pure step 6: rank pairs (by their higher-scoring member) and singles together; a pair
    costs 2 slots, a single costs 1. If only 1 slot remains on a pair's turn, skip it in favor
    of the next single; if no single remains either, inject just the pair's better half."""
    ranked = sorted(
        [("pair", a, b, sc) for a, b, sc in pairs] + [("single", iid, None, sc) for iid, sc in singles],
        key=lambda x: -x[3],
    )
    injected: list[str] = []
    budget = n_inject
    idx = 0
    while idx < len(ranked) and budget > 0:
        kind, a, b, _sc = ranked[idx]
        if kind == "single":
            injected.append(a)
            budget -= 1
            idx += 1
            continue
        if budget >= 2:
            injected.extend([a, b])
            budget -= 2
            idx += 1
        else:  # budget == 1
            next_i = next((j for j in range(idx + 1, len(ranked)) if ranked[j][0] == "single"), None)
            if next_i is not None:
                injected.append(ranked[next_i][1])
                ranked.pop(next_i)
            else:
                injected.append(a if score_map.get(a, 0) >= score_map.get(b, 0) else b)
            budget -= 1
            idx += 1
    return injected


def retrieve(query: str, now: int, task_id: str, items: dict, experiences: dict, hp,
             query_cache_path: str, desc_cache_path: str, n_inject: int):
    # 1. K_CAND candidate experiences via query-embedding cosine (append happens after scoring
    # so the current query never matches itself). The cache stores the plain embedding (so other
    # queries can later match against *this* query as a document, mirroring baseline's cache
    # convention); scoring uses the instruction-wrapped embedding, matching baseline's asymmetry
    # (memory_management.screening()) exactly -- see store.embed_query_for_scoring.
    ids, _texts, cache_emb = store.load_cache(query_cache_path)
    store.embed_and_cache(query, query_cache_path, task_id)

    if len(ids) == 0:
        return []

    scoring_vec = store.embed_query_for_scoring(query)
    cos_exp = store.cosine_to_cache(scoring_vec, cache_emb)
    scored_exp = sorted(zip(ids, cos_exp.tolist()), key=lambda x: -x[1])[: hp.k_cand]
    exp_cos = dict(scored_exp)

    # 2. expand candidate experiences' items, excluding deleted; item seen via multiple
    # experiences keeps the highest cos(q, q_exp). `ids`/exp_cos keys are raw task_ids (what
    # the query cache stores, matching run.py's embed_and_cache(query, ..., task_id) call);
    # `experiences` is keyed "ex_<task_id>" everywhere it's built (pipeline_memory.py), so the
    # lookup must re-add that prefix -- a bare `experiences.get(ex_id)` always misses and
    # silently returns zero items for every retrieval (caught by the 5.1 equivalence test).
    item_best_cos: dict[str, float] = {}
    for ex_id, cos in exp_cos.items():
        exp = experiences.get(f"ex_{ex_id}")
        if not exp:
            continue
        for iid in exp["item_ids"]:
            item = items.get(iid)
            if item is None or item["deleted"]:
                continue
            if cos > item_best_cos.get(iid, float("-inf")):
                item_best_cos[iid] = cos

    if not item_best_cos:
        return []

    # 3. score = cos(q, q_exp) + ETA * GAMMA^(now - last_active)
    score_map: dict[str, float] = {}
    for iid, cos in item_best_cos.items():
        w = hp.gamma ** max(now - items[iid]["last_active"], 0)
        score_map[iid] = cos + hp.eta * w

    # 4. group collapse: keep the member with highest query-vs-description cosine.
    desc_ids, _dtexts, desc_emb = store.load_cache(desc_cache_path)
    desc_cos_by_id = dict(zip(desc_ids, store.cosine_to_cache(scoring_vec, desc_emb).tolist())) if desc_ids else {}
    group_of = {iid: items[iid].get("group_id") for iid in item_best_cos}
    collapsed = collapse_groups(list(item_best_cos.keys()), group_of, desc_cos_by_id)

    # 5. conflict pairing (never arbitrated by recency: EFM_TASK.md 3.3.5).
    conflict_of = {iid: items[iid].get("conflict_with", []) for iid in collapsed}
    pairs, singles = pair_conflicts(collapsed, conflict_of, score_map)

    # 6. budget-limited injection.
    injected = allocate_budget(pairs, singles, n_inject, score_map)

    out = []
    for iid in injected:
        items[iid]["n_retrieved"] += 1
        out.append((iid, items[iid]))
    return out
