# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");

"""ReMe utility bookkeeping and deletion (paper 3.4, Eq. 1).

credit_outcome(): after the judge verdict for the task the items were injected into, every
injected item gets its retrieval counted (n_retrieved was already bumped at retrieval time)
and, on success, its n_success (u) bumped -- "historical utility u which increments by 1 each
time its recall contributes to a successful task completion".

prune(): phi_remove(E) = u(E)/f(E) <= beta, if f(E) >= alpha. Paper defaults alpha=5,
beta=0.5 (Appendix B.4.1). Deletion is a tombstone (`deleted: true`) rather than a dict pop,
so after a run you can still read why each experience went away -- the same post-hoc
analysis affordance efm's items.json gives.
"""

from reme.experience import log_event

ALPHA = 5      # minimum retrievals before an experience is eligible for deletion
BETA = 0.5     # success-rate threshold at or below which it is deleted


def credit_outcome(injected: list[tuple[str, dict]], judged_success: bool, now: int,
                   events_path: str, task_id) -> None:
    """Mutates items in place. `injected` is what retrieve() returned for this task."""
    for iid, item in injected:
        if judged_success:
            item["n_success"] += 1
            item["last_success"] = now
    log_event(events_path, {
        "type": "credit",
        "task_id": str(task_id),
        "judged": "success" if judged_success else "fail",
        "items": [iid for iid, _ in injected],
    })


def prune(items: dict, now: int, events_path: str, alpha: int = ALPHA,
          beta: float = BETA) -> int:
    """Deletes low-utility experiences. Returns how many were deleted this call."""
    n_deleted = 0
    for iid, item in items.items():
        if item.get("deleted"):
            continue
        f = item.get("n_retrieved", 0)
        if f < alpha:
            continue
        u = item.get("n_success", 0)
        if (u / f) <= beta:
            item["deleted"] = True
            item["deleted_reason"] = f"utility u/f={u}/{f}<={beta} with f>={alpha}"
            item["deleted_at"] = now
            n_deleted += 1
            log_event(events_path, {
                "type": "delete", "item_id": iid, "u": u, "f": f, "now": now,
            })
    return n_deleted
