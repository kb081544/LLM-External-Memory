# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");

"""ACE (Agentic Context Engineering, arXiv:2510.04618) playbook storage. Reimplementation, not
an import of github.com/ace-agent/ace -- see prompts/memory_instruction.py's ACE_REFLECTOR_SI
docstring for why. Playbook format follows the paper: a flat list of bullets, each tagged with a
section and tracking helpful/harmful vote counts, rendered as
"[id] helpful=X harmful=Y :: content" when injected into the agent prompt."""

import json
import os


def load_playbook(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_playbook(path: str, playbook: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(playbook, f, indent=2, ensure_ascii=False)


def next_bullet_id(playbook: dict) -> str:
    n = len(playbook)
    while f"b{n:04d}" in playbook:
        n += 1
    return f"b{n:04d}"


def format_playbook_for_prompt(playbook: dict, max_bullets: int = 20) -> str:
    """Rank by utility (helpful - harmful, ties broken by most helpful) and cap at
    max_bullets -- the paper's own ACE keeps the playbook bounded rather than injecting an
    ever-growing context."""
    if not playbook:
        return ""
    ranked = sorted(
        playbook.items(),
        key=lambda kv: (kv[1]["helpful"] - kv[1]["harmful"], kv[1]["helpful"]),
        reverse=True,
    )[:max_bullets]
    lines = [
        f"[{bid}] helpful={b['helpful']} harmful={b['harmful']} :: ({b['section']}) {b['content']}"
        for bid, b in ranked
    ]
    return (
        "Below is a playbook of strategies accumulated from past attempts at similar tasks "
        "(ACE-style: each entry's helpful/harmful counts reflect how often it has actually "
        "helped vs hurt so far). Use what's relevant, ignore what isn't.\n\n" + "\n".join(lines)
    )
