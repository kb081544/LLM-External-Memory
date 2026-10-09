# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");

"""CER's skills retrieval module (paper 3.2), Mind2Web port of WebArena/cer/retrieve.py.

Two differences from the WebArena version, both forced by the benchmark rather than chosen:
  - the prompt import path is Mind2Web's flat `memory_instruction` module, and
  - there is no dynamics module. CER's dynamics experiences pair a page summary with the URL
    that reaches it; Mind2Web replays already-captured steps that carry no URL, so only skills
    exist here. That is the paper's own "CER - dynamics" ablation (5.7).

Retrieval is still an LLM *selecting* ids out of the whole buffer, not embedding similarity --
that is CER's design and the point of comparison, so it is kept as-is.
"""

import re

from memory_instruction import CER_SKILLS_RETRIEVE_SI
from cer.buffer import format_skills_choices

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


def retrieve(goal: str, website: str, buf: dict, client, max_skills: int = 5) -> list[dict]:
    """Returns the selected skill entries and bumps each one's n_retrieved in place."""
    text, ids = format_skills_choices(buf)
    if not ids or max_skills <= 0:
        return []
    prompt = (
        f"Task goal: {goal}\n"
        f"Current website: {website}\n"
        f"Skills to choose from:\n{text}"
    )
    raw, _ = client.one_step_chat(
        prompt, system_msg=CER_SKILLS_RETRIEVE_SI.format(max_n=max_skills), temperature=0.0)
    picked = []
    for eid in parse_selection(raw, ids, max_skills):
        buf["skills"][eid]["n_retrieved"] += 1
        picked.append(buf["skills"][eid])
    return picked
