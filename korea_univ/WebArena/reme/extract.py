# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");

"""Parsing + prompt formatting for ReMe experiences.

The extraction prompts (prompts/memory_instruction.py REME_*) ask for blocks shaped like

    # Experience 1
    ## When to use <one sentence: the usage scenario>
    ## Content <1-3 sentences: the reusable insight>
    ## Keywords kw1, kw2, kw3
    ## Confidence high|medium|low

which is deliberately parallel to reasoningbank's Title/Description/Content block so the two
arms differ in *algorithm*, not in how much text the agent sees.
"""

import re

_ITEM_RE = re.compile(
    r"#\s*Experience\s*\d+\s*\n+"
    r"##\s*When to use\s+(?P<scenario>.*?)\n+"
    r"##\s*Content\s+(?P<content>.*?)"
    r"(?:\n+##\s*Keywords\s+(?P<keywords>.*?))?"
    r"(?:\n+##\s*Confidence\s+(?P<confidence>.*?))?"
    r"(?=\n#\s*Experience\s*\d+|\Z)",
    re.DOTALL,
)


def parse_experiences(raw_text: str) -> list[dict]:
    """Parse an LLM extraction response into experience dicts. Tolerates a fenced response
    and missing optional sections (same defensive handling as ace/curate.py)."""
    raw_text = re.sub(r"^```[a-zA-Z]*$", "", raw_text or "", flags=re.MULTILINE)
    out = []
    for m in _ITEM_RE.finditer(raw_text):
        scenario = (m.group("scenario") or "").strip()
        content = (m.group("content") or "").strip()
        if not scenario or not content:
            continue
        kw_raw = (m.group("keywords") or "").strip()
        keywords = [k.strip() for k in re.split(r"[,;]", kw_raw) if k.strip()] if kw_raw else []
        out.append({
            "scenario": scenario,
            "content": content,
            "keywords": keywords,
            "confidence": (m.group("confidence") or "").strip(),
        })
    return out


def format_item_for_prompt(number: int, item: dict) -> str:
    """Injection format. Mirrors efm/parse.py's format_item_for_prompt so every arm injects
    the same *shape* of block and the agent's existing "discuss whether you want to use each
    memory item" instruction (agents/legacy/agent.py) still applies verbatim."""
    return (
        f"# Memory Item {number}\n"
        f"## Title {item['scenario']}\n"
        f"## Description Use this when: {item['scenario']}\n"
        f"## Content {item['content']}"
    )
