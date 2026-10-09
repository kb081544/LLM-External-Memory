# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");

"""Unit tests for Mind2Web's CER port -- fake LLM client, no real API calls.
buffer.py and distill.py are byte-identical to WebArena's, so the tests that matter here are
the skills-only retrieval path and the shared parse/merge behaviour.
Run directly: python cer/test_cer_units.py
"""

import os
import sys
import tempfile

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


print("distill.parse_skills / merge_skills")
from cer.distill import parse_skills, merge_skills

raw = """<<think>>
break into sub-goals
<</think>>
<<skill>>
Navigate to category {category name}.
<</skill>>
<<steps>>
1. Click the department menu.
```click(dept id)```
2. Click the sub-category link.
```click(subcategory id)```
<</steps>>
<<think>>
next
<</think>>
<<skill>>
Add {item type} to cart
<</skill>>
<<steps>>
1. Open the product page.
2. Click "Add to Cart".
<</steps>>"""
skills = parse_skills(raw)
check("parses 2 skills", len(skills) == 2)
check("variable placeholder kept", "{category name}" in skills[0]["name"])
check("think text not stored", all("sub-goals" not in s["steps"] for s in skills))
check("'Summarized before' skipped", parse_skills(
    "<<skill>>\nX\n<</skill>>\n<<steps>>\nSummarized before\n<</steps>>") == [])

buf = {"dynamics": {}, "skills": {}}
added = merge_skills(skills, buf, task_id="ann-1", now=0)
check("both skills merged", len(added) == 2 and len(buf["skills"]) == 2)
check("dedup by normalized name", merge_skills(
    [{"name": "navigate to category {CATEGORY NAME}", "steps": "1. x"}], buf, "ann-2", 1) == [])
check("annotation_id recorded as source", buf["skills"]["s0000"]["source_task"] == "ann-1")


print("buffer rendering (skills-only)")
from cer.buffer import (
    load_buffer, save_buffer, format_existing_skills, format_skills_choices, render_for_replay,
)
from memory_instruction import CER_REPLAY_INSTRUCTION

menu, ids = format_skills_choices(buf)
check("menu numbered from 1", menu.startswith("Skill 1:") and len(ids) == 2)
check("existing-skills block built", "Skill 1:" in format_existing_skills(buf))
replay = render_for_replay([], list(buf["skills"].values()), CER_REPLAY_INSTRUCTION)
check("replay has skills section", "## Useful skills" in replay)
check("replay has no pages section (skills-only arm)", "## Useful pages" not in replay)
check("cold buffer replays nothing", render_for_replay([], [], CER_REPLAY_INSTRUCTION) == "")

with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "sub", "buffer.json")
    save_buffer(p, buf)
    check("buffer round-trips", load_buffer(p)["skills"].keys() == buf["skills"].keys())


print("retrieve.retrieve (fake LLM client)")
from cer.retrieve import parse_selection, retrieve

check("ids parsed from selected block", parse_selection(
    "<<selected-skills>>\nid: 2; name: X\n<</selected-skills>>", ["s0", "s1"], 5) == ["s1"])
check("out-of-range dropped", parse_selection("id: 9", ["s0"], 5) == [])
check("max_n truncates", len(parse_selection("id: 1\nid: 2", ["a", "b"], 1)) == 1)


class FakeClient:
    def __init__(self):
        self.calls = []

    def one_step_chat(self, text, system_msg=None, json_mode=False, temperature=0.0):
        self.calls.append({"text": text, "system_msg": system_msg})
        return "<<selected-skills>>\nid: 1; name: whatever\n<</selected-skills>>", None


client = FakeClient()
picked = retrieve("buy a thing", "kohls", buf, client, max_skills=5)
check("exactly one LLM call (no dynamics module)", len(client.calls) == 1)
check("one skill selected", len(picked) == 1)
check("n_retrieved bumped", sorted(s["n_retrieved"] for s in buf["skills"].values()) == [0, 1])
check("k substituted into prompt", "not more than 5" in client.calls[0]["system_msg"])
check("goal and website passed", "buy a thing" in client.calls[0]["text"]
      and "kohls" in client.calls[0]["text"])

cold = FakeClient()
check("cold buffer makes no call", retrieve("g", "w", {"dynamics": {}, "skills": {}}, cold) == []
      and cold.calls == [])


print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
