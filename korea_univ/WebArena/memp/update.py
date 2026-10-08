# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");

"""Memp memory update strategies (paper 4.3). The pipeline picks one with --memp_update:

  vanilla     -- "After every t tasks, all trajectories from these tasks are consolidated into
                 procedural memories and directly appended to the memory bank." Every task's
                 built procedure is appended, success or failure.
  validation  -- only trajectories the judge called a success are consolidated and appended.
  adjustment  -- success is appended as in validation; on failure the procedure that was
                 actually injected (rank 1) is revised IN PLACE from the failed trajectory
                 (MEMP_ADJUST_SI), which is Memp's one genuinely destructive edit and the
                 behaviour Useful Memories (arXiv 2026) singles out as the most damaging step
                 -- keeping it faithful here is the point of having this arm.

`t` (the paper's update period) is --memp_update_every; with t=1 the update runs after every
task, which is what our one-trajectory-per-task pipeline does by default.
"""

from memp.procedure import log_event, next_id, parse_procedure
from prompts.memory_instruction import MEMP_ADJUST_SI


def format_existing_for_adjust(proc: dict, query: str, trajectory: str) -> str:
    steps = "\n".join(proc.get("steps", []))
    return (
        f"## EXISTING PROCEDURE\n# Procedure\n## When to use {proc.get('when_to_use','')}\n"
        f"## Steps\n{steps}\n\n"
        f"## FAILED ATTEMPT\n**Query:** {query}\n\n**Trajectory:**\n{trajectory}"
    )


def apply_update(mode: str, procs: dict, parsed: dict | None, *, judged_success: bool,
                 injected: list[tuple[str, dict]], query: str, trajectory: str, task_id,
                 now: int, client, events_path: str) -> tuple[str | None, bool]:
    """Mutates `procs` in place.

    Returns (new_proc_id_or_None, adjusted_flag):
      new_proc_id  -- id of a newly appended procedure, if this call appended one
      adjusted     -- True if an existing procedure was revised in place
    """
    if mode == "vanilla":
        return _append(procs, parsed, query, trajectory, task_id, now,
                       judged_success, events_path), False

    if mode == "validation":
        if judged_success:
            return _append(procs, parsed, query, trajectory, task_id, now,
                           judged_success, events_path), False
        log_event(events_path, {"type": "validation_skip", "task_id": str(task_id)})
        return None, False

    if mode == "adjustment":
        if judged_success:
            return _append(procs, parsed, query, trajectory, task_id, now,
                           judged_success, events_path), False
        if not injected:
            # Nothing was injected (cold bank) -- there is no procedure to blame, so this
            # degrades to validation's behaviour rather than inventing an edit target.
            log_event(events_path, {"type": "adjust_skip_no_injection", "task_id": str(task_id)})
            return None, False
        target_id, target = injected[0]
        text, _ = client.one_step_chat(
            format_existing_for_adjust(target, query, trajectory),
            system_msg=MEMP_ADJUST_SI, temperature=0.7,
        )
        revised = parse_procedure(text or "")
        if not revised:
            log_event(events_path, {"type": "adjust_unparsed", "task_id": str(task_id),
                                    "item_id": target_id})
            return None, False
        target["when_to_use"] = revised["when_to_use"]
        target["steps"] = revised["steps"]
        target["n_revisions"] = target.get("n_revisions", 0) + 1
        target["last_revised"] = now
        target.setdefault("revised_by", []).append(str(task_id))
        log_event(events_path, {"type": "adjust", "task_id": str(task_id),
                                "item_id": target_id, "n_revisions": target["n_revisions"]})
        return None, True

    raise ValueError(f"apply_update: unknown mode {mode!r}")


def _append(procs: dict, parsed: dict | None, query: str, trajectory: str, task_id, now: int,
            judged_success: bool, events_path: str) -> str | None:
    if not parsed:
        log_event(events_path, {"type": "build_unparsed", "task_id": str(task_id)})
        return None
    pid = next_id(procs)
    procs[pid] = {
        "when_to_use": parsed["when_to_use"],
        "steps": parsed["steps"],
        "trajectory": trajectory,
        "source_task": str(task_id),
        "source_query": query,
        "source_status": "success" if judged_success else "fail",
        "created": now,
        "n_retrieved": 0,
        "n_revisions": 0,
        "deleted": False,
    }
    log_event(events_path, {"type": "append", "task_id": str(task_id), "item_id": pid,
                            "status": "success" if judged_success else "fail"})
    return pid
