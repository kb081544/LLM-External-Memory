# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");

"""EFM_TASK.md 3.6 -- forgetting. Runs after usage recording each task. Rule 3 ("제한적
품질 삭제") is off by default (hp.quality_delete=False) per the doc; the hook exists so turning
it on is a one-line hparam flip, but it is not exercised by the default experiment."""

from efm.usage import _log_event


def forget(items: dict, now: int, hp, events_path: str) -> int:
    newly_deleted = []
    for item_id, item in items.items():
        if item["deleted"] or item["pinned"]:
            continue
        if now - item["created_at"] < hp.grace:
            continue
        if now - item["last_active"] > hp.t_stale:
            item["deleted"] = True
            item["deleted_reason"] = "stale"
        elif item["n_retrieved"] >= hp.r_min and item["n_used"] / item["n_retrieved"] < hp.rho:
            item["deleted"] = True
            item["deleted_reason"] = "declined"
        # hp.quality_delete rule (n_used>=5, last 3 used-tasks all judge-failed) intentionally
        # not evaluated here while off by default -- see module docstring.
        if item["deleted"]:
            newly_deleted.append(item_id)

    if not newly_deleted:
        return 0

    deleted_set = set(newly_deleted)
    for item_id, item in items.items():
        if item_id in deleted_set:
            continue
        kept = []
        for other in item.get("conflict_with", []):
            if other in deleted_set:
                _log_event(events_path, {
                    "type": "conflict_resolved",
                    "kept": item_id,
                    "deleted": other,
                    "reason": items[other]["deleted_reason"],
                    "kept_is_newer": item["created_at"] > items[other]["created_at"],
                })
            else:
                kept.append(other)
        item["conflict_with"] = kept

    group_members: dict[str, list[str]] = {}
    for item_id, item in items.items():
        if item["deleted"]:
            continue
        gid = item.get("group_id")
        if gid:
            group_members.setdefault(gid, []).append(item_id)
    for members in group_members.values():
        if len(members) <= 1:
            for m in members:
                items[m]["group_id"] = None

    return len(newly_deleted)
