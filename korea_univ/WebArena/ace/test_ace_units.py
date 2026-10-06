# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");

"""Unit tests for ace/curate.py and ace/playbook.py -- fake embeddings, no real API calls.
Run directly: python ace/test_ace_units.py
"""

import os
import sys
import tempfile
from unittest.mock import patch

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

passed = failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok   {name}")
    else:
        failed += 1
        print(f"  FAIL {name}")


# ---------------------------------------------------------------- curate.parse_candidate_bullets
print("curate.parse_candidate_bullets")
from ace.curate import parse_candidate_bullets

raw = "[strategy] Always confirm the quantity field before submitting.\n\n[mistake] Do not click disabled checkout buttons."
bullets = parse_candidate_bullets(raw)
check("parses 2 bullets", len(bullets) == 2)
check("bullet 1 section", bullets[0]["section"] == "strategy")
check("bullet 1 content", "quantity field" in bullets[0]["content"])
check("bullet 2 section", bullets[1]["section"] == "mistake")

check("ignores malformed block", len(parse_candidate_bullets("no brackets here")) == 0)


# ---------------------------------------------------------------- playbook.format_playbook_for_prompt
print("playbook.format_playbook_for_prompt")
from ace.playbook import format_playbook_for_prompt, next_bullet_id

pb = {
    "b0000": {"section": "strategy", "content": "low utility", "helpful": 1, "harmful": 5},
    "b0001": {"section": "mistake", "content": "high utility", "helpful": 5, "harmful": 0},
}
block = format_playbook_for_prompt(pb, max_bullets=20)
check("empty playbook renders empty string", format_playbook_for_prompt({}) == "")
check("higher-utility bullet ranked first", block.index("high utility") < block.index("low utility"))
check("next_bullet_id avoids collision", next_bullet_id(pb) == "b0002")
check("cap respected", len(parse_candidate_bullets(
    "\n\n".join(f"[strategy] s{i}" for i in range(5))
)) == 5)


# ---------------------------------------------------------------- curate.curate (mocked embeddings)
print("curate.curate (mocked embeddings)")


def fake_embed(text, dimensionality=3072):
    """SAME_AS marker -> dim 0 (forces a dedup match); anything else -> its own dim via hash,
    so unrelated bullets never collide."""
    vec = torch.zeros(1, 4)
    if "SAME_AS_EXISTING" in text:
        vec[0, 0] = 1.0
    else:
        vec[0, (abs(hash(text)) % 3) + 1] = 1.0
    return vec


with tempfile.TemporaryDirectory() as td, patch("ace.curate.embed_query_with_gemini", side_effect=fake_embed):
    from ace.curate import curate

    cache_path = os.path.join(td, "content.jsonl")

    # First candidate: nothing in the playbook yet -> always a new bullet.
    playbook = {}
    added = curate(
        [{"section": "strategy", "content": "SAME_AS_EXISTING baseline strategy"}],
        playbook, task_succeeded=True, cache_path=cache_path,
    )
    check("first bullet added as new", added == 1 and len(playbook) == 1)
    first_id = next(iter(playbook))
    check("new bullet starts helpful=1 harmful=0 on success", playbook[first_id]["helpful"] == 1 and playbook[first_id]["harmful"] == 0)

    # Second candidate: embeds to the same dim (SAME_AS_EXISTING) -> should match, not duplicate.
    added = curate(
        [{"section": "strategy", "content": "SAME_AS_EXISTING but phrased differently"}],
        playbook, task_succeeded=True, cache_path=cache_path,
    )
    check("semantic duplicate merged, not added", added == 0 and len(playbook) == 1)
    check("merge bumps helpful on success", playbook[first_id]["helpful"] == 2)

    # Third candidate: same dedup match, but this time the task failed -> bumps harmful.
    added = curate(
        [{"section": "strategy", "content": "SAME_AS_EXISTING once more"}],
        playbook, task_succeeded=False, cache_path=cache_path,
    )
    check("merge bumps harmful on failure", playbook[first_id]["harmful"] == 1)

    # Fourth candidate: genuinely different content -> new bullet, playbook now has 2.
    added = curate(
        [{"section": "mistake", "content": "a totally unrelated pitfall to avoid"}],
        playbook, task_succeeded=False, cache_path=cache_path,
    )
    check("unrelated content added as new bullet", added == 1 and len(playbook) == 2)


print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
