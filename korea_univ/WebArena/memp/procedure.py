# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");

"""Memp (Exploring Agent Procedural Memory, Findings of ACL 2026) procedure store.

Reimplementation, not an import of the released ALFWorld/TravelPlanner code (see
prompts/memory_instruction.py's MEMP_BUILD_SI docstring).

A procedure is the paper's memory unit m_p = B(tau, r):
  when_to_use -- the task class this procedure covers (also the retrieval key, "Key=Query"
                 in the paper's retrieval ablation is the source task's query; we index the
                 source query for exactly that reason -- see memp/retrieve.py)
  steps       -- the generalized ordered steps (BUILD = "script")
  trajectory  -- the raw source trajectory text (BUILD = "trajectory")
Both fields are stored always; the --memp_build flag decides which of them is injected, which
is how the paper's three BUILD conditions (script / trajectory / proceduralization) are
expressed without re-running extraction three times.
"""

import json
import os
import re

_PROC_RE = re.compile(
    r"#\s*Procedure\s*\n+"
    r"##\s*When to use\s+(?P<when>.*?)\n+"
    r"##\s*Steps\s+(?P<steps>.*?)\Z",
    re.DOTALL,
)


def load_procedures(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_procedures(path: str, procs: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(procs, f, indent=2, ensure_ascii=False)


def next_id(procs: dict) -> str:
    n = len(procs)
    while f"p{n:04d}" in procs:
        n += 1
    return f"p{n:04d}"


def parse_procedure(raw_text: str) -> dict | None:
    """Parse MEMP_BUILD_SI / MEMP_ADJUST_SI output. Returns None if the response has no
    recognizable procedure block (caller then skips storing anything for this task)."""
    raw_text = re.sub(r"^```[a-zA-Z]*$", "", raw_text or "", flags=re.MULTILINE).strip()
    m = _PROC_RE.search(raw_text)
    if not m:
        return None
    when = m.group("when").strip()
    steps = [s.strip() for s in m.group("steps").strip().splitlines() if s.strip()]
    if not when or not steps:
        return None
    return {"when_to_use": when, "steps": steps}


def format_for_prompt(number: int, proc: dict, build: str) -> str:
    """Injection block. `build` selects what the paper's BUILD condition shows the agent:
      script            -- the generalized steps only
      trajectory        -- the raw source trajectory only
      proceduralization -- both ("combines the full retrieved trajectories with the
                           high-level script", paper 4.2)
    Wrapped in the same "# Memory Item N / ## Title / ## Description / ## Content" envelope as
    every other arm so the agent-side prompt shape is identical across arms."""
    body_parts = []
    if build in ("script", "proceduralization"):
        body_parts.append("Steps:\n" + "\n".join(proc.get("steps", [])))
    if build in ("trajectory", "proceduralization"):
        traj = (proc.get("trajectory") or "").strip()
        if traj:
            body_parts.append("Source trajectory:\n" + traj)
    body = "\n\n".join(body_parts) if body_parts else "\n".join(proc.get("steps", []))
    return (
        f"# Memory Item {number}\n"
        f"## Title {proc.get('when_to_use', 'Procedure')}\n"
        f"## Description Use this procedure when: {proc.get('when_to_use', '')}\n"
        f"## Content {body}"
    )


def log_event(events_path: str, event: dict) -> None:
    os.makedirs(os.path.dirname(events_path) or ".", exist_ok=True)
    with open(events_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
