# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");

"""Parsing + merging for CER's two distillation modules (paper 3.1).

The prompts (prompts/memory_instruction.py, transcribed from the paper's Fig. 3/4) ask for
`<<URL>>/<<think>>/<<page-summary>>` triples and `<<think>>/<<skill>>/<<steps>>` triples. The
`<<think>>` blocks are ReAct-style scratch space and are deliberately dropped here -- only the
summary/skill payload is stored.

Dedup is the paper's own prompt-level mechanism (the model is shown the existing buffer and
answers "Summarized before"), but a model that ignores that instruction would otherwise grow
the buffer without bound, so the merge functions also drop an entry whose URL (dynamics, the
prompt's own uniqueness key) or normalized name (skills) is already present. No LLM or
embedding call happens here.
"""

import re

_BLOCK = r"<<{tag}>>(.*?)<</{tag}>>"
_URL_RE = re.compile(_BLOCK.format(tag="URL"), re.DOTALL)
_PAGE_RE = re.compile(_BLOCK.format(tag="page-summary"), re.DOTALL)
_SKILL_RE = re.compile(_BLOCK.format(tag="skill"), re.DOTALL)
_STEPS_RE = re.compile(_BLOCK.format(tag="steps"), re.DOTALL)

_SUMMARIZED_BEFORE = "summarized before"


def _field(text: str, *labels: str) -> str:
    """Pull "Label: value" out of a page summary. The paper's format block puts each field on
    its own line while its worked example separates them with "; ", so accept either."""
    for label in labels:
        m = re.search(rf"{label}\s*:\s*(.+?)(?:\n|;|$)", text, re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip(".;").strip()
    return ""


def parse_dynamics(raw: str) -> list[dict]:
    urls = _URL_RE.findall(raw or "")
    pages = _PAGE_RE.findall(raw or "")
    out = []
    for url, page in zip(urls, pages):
        url, page = url.strip(), page.strip()
        if not url or _SUMMARIZED_BEFORE in page.lower():
            continue
        name = _field(page, "Name")
        if not name:
            continue
        out.append({
            "url": url,
            "name": name,
            "description": _field(page, "Description"),
            "usages": _field(page, "Possible usages", "Usages"),
        })
    return out


def parse_skills(raw: str) -> list[dict]:
    names = _SKILL_RE.findall(raw or "")
    steps = _STEPS_RE.findall(raw or "")
    out = []
    for name, step_text in zip(names, steps):
        name, step_text = name.strip().rstrip("."), step_text.strip()
        if not name or not step_text or _SUMMARIZED_BEFORE in step_text.lower():
            continue
        out.append({"name": name, "steps": step_text})
    return out


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def merge_dynamics(entries: list[dict], buf: dict, task_id: str, now: int) -> list[str]:
    from cer.buffer import next_id

    section = buf["dynamics"]
    seen_urls = {d["url"] for d in section.values()}
    added = []
    for e in entries:
        if e["url"] in seen_urls:
            continue
        eid = next_id(section, "d")
        section[eid] = {
            "name": e["name"],
            "description": e["description"],
            "usages": e["usages"],
            "url": e["url"],
            "source_task": task_id,
            "created": now,
            "n_retrieved": 0,
        }
        seen_urls.add(e["url"])
        added.append(eid)
    return added


def merge_skills(entries: list[dict], buf: dict, task_id: str, now: int) -> list[str]:
    from cer.buffer import next_id

    section = buf["skills"]
    seen_names = {_norm(s["name"]) for s in section.values()}
    added = []
    for e in entries:
        key = _norm(e["name"])
        if key in seen_names:
            continue
        eid = next_id(section, "s")
        section[eid] = {
            "name": e["name"],
            "steps": e["steps"],
            "source_task": task_id,
            "created": now,
            "n_retrieved": 0,
        }
        seen_names.add(key)
        added.append(eid)
    return added
