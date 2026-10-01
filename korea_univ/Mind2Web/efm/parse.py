# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");

"""Parses the *unmodified* SUCCESSFUL_SI/FAILED_SI extraction output (memory_instruction.py,
shared verbatim with WebArena) into EFM's title/description/content item schema. Pure string
parsing -- no LLM call, and the extraction prompt itself is never touched. Ported unchanged from
WebArena/efm/parse.py -- this logic is domain-agnostic."""

import re

_ITEM_RE = re.compile(
    r"#\s*Memory Item\s*\d+\s*\n+"
    r"##\s*Title\s+(?P<title>.*?)\n+"
    r"##\s*Description\s+(?P<description>.*?)\n+"
    r"##\s*Content\s+(?P<content>.*?)"
    r"(?=\n#\s*Memory Item\s*\d+|\Z)",
    re.DOTALL,
)


def parse_memory_items(memory_items: list[str]) -> list[dict]:
    """memory_items: the raw list already stored in a baseline jsonl line (each element is
    roughly one "# Memory Item i..." block). Re-joining and re-parsing with a proper regex is
    more robust than trusting any upstream split boundary landed exactly on item boundaries."""
    blob = "\n\n".join(memory_items)
    items = []
    for m in _ITEM_RE.finditer(blob):
        title = m.group("title").strip()
        description = m.group("description").strip()
        content = m.group("content").strip()
        if not (title or description or content):
            continue
        items.append({"title": title, "description": description, "content": content})
    return items


def format_item_for_prompt(number: int, item: dict) -> str:
    """Injection format: identical Title+Description+Content shape to baseline's raw
    memory_items injection (memory.py's format_memory_for_prompt() re-emits the extracted
    blocks verbatim) -- EFM must match this exactly for a same-conditions comparison."""
    return (
        f"# Memory Item {number}\n"
        f"## Title {item['title']}\n"
        f"## Description {item['description']}\n"
        f"## Content {item['content']}"
    )
