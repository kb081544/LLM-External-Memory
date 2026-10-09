# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");

"""CER's two retrieval modules (paper 3.2).

Unlike every other arm in this repo, retrieval here is **not** embedding similarity: the model
is shown the whole buffer as a numbered menu and asked to pick at most k entries for the
current goal (prompts Fig. 5/6). That is CER's design and the main structural difference worth
comparing against, so it is kept rather than swapped for the shared embedding retriever --
which also means retrieval costs one LLM call per section per task.
"""

import re

from prompts.memory_instruction import (
    CER_DYNAMICS_RETRIEVE_SI,
    CER_SKILLS_RETRIEVE_SI,
)
from cer.buffer import format_dynamics_choices, format_skills_choices

_ID_RE = re.compile(r"id\s*:\s*(\d+)", re.IGNORECASE)
_SELECTED_RE = re.compile(r"<<selected-(?:pages|skills)>>(.*?)<</selected-(?:pages|skills)>>", re.DOTALL)


def parse_selection(raw: str, ids: list[str], max_n: int) -> list[str]:
    """Map the model's 1-based menu ids back to buffer ids. Out-of-range or duplicate ids are
    dropped rather than raising -- a malformed pick should cost us an experience, not the task."""
    body = raw or ""
    m = _SELECTED_RE.search(body)
    if m:
        body = m.group(1)
    picked, seen = [], set()
    for token in _ID_RE.findall(body):
        idx = int(token)
        if 1 <= idx <= len(ids):
            eid = ids[idx - 1]
            if eid not in seen:
                seen.add(eid)
                picked.append(eid)
    return picked[:max_n]


def _select(section_text: str, ids: list[str], goal: str, website: str, system_msg: str,
            label: str, max_n: int, client) -> list[str]:
    if not ids:
        return []
    # Field names mirror the prompts' own worked examples ("Shortcuts to choose from:" for
    # pages, "Skills to choose from:" for skills) so the menu looks like what they demonstrate.
    prompt = (
        f"Task goal: {goal}\n"
        f"Current website: {website}\n"
        f"{label} to choose from:\n"
        f"{section_text}"
    )
    raw, _ = client.one_step_chat(prompt, system_msg=system_msg.format(max_n=max_n), temperature=0.0)
    return parse_selection(raw, ids, max_n)


def retrieve(goal: str, website: str, buf: dict, client, max_dynamics: int = 5,
             max_skills: int = 5, use_dynamics: bool = True) -> tuple[list[dict], list[dict]]:
    """Returns (selected_dynamics, selected_skills) as entry dicts, and bumps each selected
    entry's n_retrieved in place. `use_dynamics=False` runs the paper's "CER - dynamics"
    ablation (5.7), which is what Mind2Web needs since its offline steps carry no URL."""
    picked_dyn: list[dict] = []
    if use_dynamics and max_dynamics > 0:
        text, ids = format_dynamics_choices(buf)
        for eid in _select(text, ids, goal, website, CER_DYNAMICS_RETRIEVE_SI,
                           "Shortcuts", max_dynamics, client):
            buf["dynamics"][eid]["n_retrieved"] += 1
            picked_dyn.append(buf["dynamics"][eid])

    picked_skills: list[dict] = []
    if max_skills > 0:
        text, ids = format_skills_choices(buf)
        for eid in _select(text, ids, goal, website, CER_SKILLS_RETRIEVE_SI,
                           "Skills", max_skills, client):
            buf["skills"][eid]["n_retrieved"] += 1
            picked_skills.append(buf["skills"][eid])

    return picked_dyn, picked_skills
