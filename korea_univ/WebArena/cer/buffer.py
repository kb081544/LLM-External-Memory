# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");

"""CER's experience buffer (arXiv:2506.06698, paper 3.1-3.3).

An experience is a pair E = (D, S): `dynamics` entries describe a page and the URL that
reaches it, `skills` entries describe a reusable sub-goal procedure. They are kept in two
separate sections because CER distills and retrieves them with separate modules.

This module is only storage plus the text renderings the prompts need:
  - `format_existing_*` feeds the distillation prompts their "Existing ..." block, which is
    how CER avoids re-distilling what it already knows (prompt notes 2-4 in Fig. 3/4).
  - `format_*_choices` renders the numbered menu the retrieval prompts select ids out of
    (Fig. 5/6), so the ids the model answers with are 1-based positions in that menu.
  - `render_for_replay` is the paper's f mapping (3.3): selected experiences -> natural
    language pasted into the agent's context.
"""

import json
import os

SECTIONS = ("dynamics", "skills")


def load_buffer(path: str) -> dict:
    if not os.path.exists(path):
        return {"dynamics": {}, "skills": {}}
    with open(path, "r", encoding="utf-8") as f:
        buf = json.load(f)
    for section in SECTIONS:
        buf.setdefault(section, {})
    return buf


def save_buffer(path: str, buf: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(buf, f, indent=2, ensure_ascii=False)


def next_id(section: dict, prefix: str) -> str:
    n = len(section)
    while f"{prefix}{n:04d}" in section:
        n += 1
    return f"{prefix}{n:04d}"


def _ordered(section: dict) -> list[tuple[str, dict]]:
    """Insertion order is the buffer's own id order, which keeps the retrieval menu's 1-based
    ids stable for a given buffer state."""
    return sorted(section.items(), key=lambda kv: kv[0])


def format_existing_dynamics(buf: dict) -> str:
    rows = _ordered(buf["dynamics"])
    if not rows:
        return "(none yet)"
    out = []
    for i, (_eid, d) in enumerate(rows, 1):
        out.append(
            f"Page {i}: {d['name']}\n"
            f"Description: {d['description']}\n"
            f"Usages: {d['usages']}\n"
            f"URL: {d['url']}"
        )
    return "\n".join(out)


def format_existing_skills(buf: dict) -> str:
    rows = _ordered(buf["skills"])
    if not rows:
        return "(none yet)"
    out = []
    for i, (_eid, s) in enumerate(rows, 1):
        out.append(f"Skill {i}: {s['name']}\n{s['steps']}")
    return "\n".join(out)


def format_dynamics_choices(buf: dict) -> tuple[str, list[str]]:
    """Returns (menu_text, ids) where ids[i] is the entry the model's "id: i+1" refers to."""
    rows = _ordered(buf["dynamics"])
    lines, ids = [], []
    for i, (eid, d) in enumerate(rows, 1):
        ids.append(eid)
        lines.append(
            f"id: {i}; name: {d['name']}; description: {d['description']}; "
            f"possible usages: {d['usages']}; url: {d['url']}"
        )
    return "\n".join(lines), ids


def format_skills_choices(buf: dict) -> tuple[str, list[str]]:
    rows = _ordered(buf["skills"])
    lines, ids = [], []
    for i, (eid, s) in enumerate(rows, 1):
        ids.append(eid)
        lines.append(f"Skill {i}: {s['name']}\n{s['steps']}")
    return "\n".join(lines), ids


def render_for_replay(dynamics: list[dict], skills: list[dict], header: str) -> str:
    """The paper's f mapping (3.3). Returns "" when nothing was selected so a cold buffer
    injects no scaffolding text at all."""
    if not dynamics and not skills:
        return ""
    parts = [header]
    if dynamics:
        block = ["## Useful pages"]
        for i, d in enumerate(dynamics, 1):
            block.append(
                f"{i}. {d['name']} -- {d['description']} "
                f"Usages: {d['usages']} URL: {d['url']}"
            )
        parts.append("\n".join(block))
    if skills:
        block = ["## Useful skills"]
        for i, s in enumerate(skills, 1):
            block.append(f"{i}. {s['name']}\n{s['steps']}")
        parts.append("\n".join(block))
    return "\n\n".join(parts)
