# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");

"""Unit tests for cer/{buffer,distill,retrieve}.py -- fake LLM client, no real API calls.
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


# ---------------------------------------------------------------- distill.parse_dynamics
print("distill.parse_dynamics")
from cer.distill import parse_dynamics, parse_skills, merge_dynamics, merge_skills

raw_dyn = """<<URL>>
http://site/forums
<</URL>>
<<think>>
scratch reasoning that must not be stored
<</think>>
<<page-summary>>
Name: Forums page; Description: it shows a list of forums.; Possible usages: navigate to forums.
<</page-summary>>
<<URL>>
http://site/profile
<</URL>>
<<think>>
more scratch
<</think>>
<<page-summary>>
Name: Profile page
Description: shows the user's profile
Usages: view or edit the profile
<</page-summary>>"""
dyn = parse_dynamics(raw_dyn)
check("parses 2 pages", len(dyn) == 2)
check("semicolon-separated fields parse", dyn[0]["name"] == "Forums page" and "list of forums" in dyn[0]["description"])
check("newline-separated fields parse", dyn[1]["name"] == "Profile page" and dyn[1]["usages"].startswith("view or edit"))
check("url captured", dyn[0]["url"] == "http://site/forums")
check("think block dropped", all("scratch" not in str(d) for d in dyn))

skipped = parse_dynamics("""<<URL>>
http://site/forums
<</URL>>
<<page-summary>>
Summarized before
<</page-summary>>""")
check("'Summarized before' page skipped", skipped == [])
check("garbage yields nothing", parse_dynamics("no tags at all") == [])


# ---------------------------------------------------------------- distill.parse_skills
print("distill.parse_skills")
raw_skills = """<<think>>
break the goal into sub-goals
<</think>>
<<skill>>
Navigate to forum {forum name}.
<</skill>>
<<steps>>
1. Click the "Forums" menu item.
```click(forums id)```
2. Click the forum name.
```click(forum name id)```
<</steps>>
<<think>>
second sub-goal
<</think>>
<<skill>>
Sort posts by {criterion}
<</skill>>
<<steps>>
1. Click "Sort by".
```click(sort by id)```
<</steps>>"""
skills = parse_skills(raw_skills)
check("parses 2 skills", len(skills) == 2)
check("trailing period trimmed from name", skills[0]["name"] == "Navigate to forum {forum name}")
check("steps kept verbatim", "click(forums id)" in skills[0]["steps"])
check("variable placeholders preserved", "{criterion}" in skills[1]["name"])
check("'Summarized before' skill skipped", parse_skills(
    "<<skill>>\nX\n<</skill>>\n<<steps>>\nSummarized before\n<</steps>>") == [])


# ---------------------------------------------------------------- distill merge/dedup
print("distill.merge_* (dedup)")
buf = {"dynamics": {}, "skills": {}}
added = merge_dynamics(dyn, buf, task_id="t1", now=0)
check("both pages added to empty buffer", len(added) == 2 and len(buf["dynamics"]) == 2)
check("new dynamics entry starts at n_retrieved=0", all(d["n_retrieved"] == 0 for d in buf["dynamics"].values()))
again = merge_dynamics(dyn, buf, task_id="t2", now=1)
check("same URLs not added twice", again == [] and len(buf["dynamics"]) == 2)

added_s = merge_skills(skills, buf, task_id="t1", now=0)
check("both skills added", len(added_s) == 2 and len(buf["skills"]) == 2)
dupe_name = merge_skills([{"name": "navigate to forum {FORUM NAME}!", "steps": "1. x"}], buf, "t2", 1)
check("skill dedup is name-normalized", dupe_name == [] and len(buf["skills"]) == 2)
check("source task recorded", buf["skills"][sorted(buf["skills"])[0]]["source_task"] == "t1")


# ---------------------------------------------------------------- buffer rendering
print("buffer rendering")
from cer.buffer import (
    load_buffer, save_buffer, format_existing_dynamics, format_existing_skills,
    format_dynamics_choices, format_skills_choices, render_for_replay, next_id,
)
from prompts.memory_instruction import CER_REPLAY_INSTRUCTION

menu, ids = format_dynamics_choices(buf)
check("dynamics menu is 1-based and ordered", menu.startswith("id: 1;") and len(ids) == 2)
check("dynamics menu carries the url", "http://site/forums" in menu)
smenu, sids = format_skills_choices(buf)
check("skills menu numbered Skill 1..n", smenu.startswith("Skill 1:") and len(sids) == 2)
check("existing-dynamics block lists pages", "Page 1:" in format_existing_dynamics(buf))
check("existing-skills block lists skills", "Skill 1:" in format_existing_skills(buf))
check("empty buffer renders a placeholder", format_existing_dynamics({"dynamics": {}, "skills": {}}) == "(none yet)")
check("next_id avoids collision", next_id(buf["dynamics"], "d") == "d0002")

replay = render_for_replay(list(buf["dynamics"].values()), list(buf["skills"].values()), CER_REPLAY_INSTRUCTION)
check("replay block has both sections", "## Useful pages" in replay and "## Useful skills" in replay)
check("cold buffer replays nothing", render_for_replay([], [], CER_REPLAY_INSTRUCTION) == "")

with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "nested", "buffer.json")
    save_buffer(p, buf)
    check("buffer round-trips", load_buffer(p)["skills"].keys() == buf["skills"].keys())
    check("missing buffer loads as empty", load_buffer(os.path.join(td, "nope.json")) == {"dynamics": {}, "skills": {}})


# ---------------------------------------------------------------- retrieve (fake client)
print("retrieve.retrieve (fake LLM client)")
from cer.retrieve import parse_selection, retrieve

check("parses ids inside the selected block", parse_selection(
    "<<think>>noise id: 99<</think>>\n<<selected-pages>>\nid: 2; name: X\n<</selected-pages>>",
    ["d0", "d1", "d2"], 5) == ["d1"])
check("out-of-range ids dropped", parse_selection("<<selected-skills>>id: 7<</selected-skills>>", ["s0"], 5) == [])
check("duplicate ids collapsed", parse_selection("id: 1\nid: 1", ["s0", "s1"], 5) == ["s0"])
check("max_n truncates", len(parse_selection("id: 1\nid: 2\nid: 3", ["a", "b", "c"], 2)) == 2)
check("no ids -> nothing selected", parse_selection("<<selected-skills>>none<</selected-skills>>", ["s0"], 5) == [])


class FakeClient:
    """Answers every retrieval call with 'id: 1', and records which system messages it saw."""

    def __init__(self):
        self.calls = []

    def one_step_chat(self, text, system_msg=None, json_mode=False, temperature=0.0):
        self.calls.append({"text": text, "system_msg": system_msg})
        return "<<think>>t<</think>>\n<<selected-pages>>\nid: 1; name: whatever\n<</selected-pages>>", None


client = FakeClient()
picked_d, picked_s = retrieve("some goal", "shopping", buf, client, max_dynamics=5, max_skills=5)
check("one call per section", len(client.calls) == 2)
check("dynamics selected", len(picked_d) == 1)
check("skills selected", len(picked_s) == 1)
check("n_retrieved bumped on the picked dynamics", sorted(
    d["n_retrieved"] for d in buf["dynamics"].values()) == [0, 1])
check("max_n substituted into the prompt", "not more than 5" in client.calls[0]["system_msg"])
check("goal passed to the model", "some goal" in client.calls[0]["text"])

client2 = FakeClient()
picked_d2, picked_s2 = retrieve("g", "w", buf, client2, use_dynamics=False)
check("use_dynamics=False skips the dynamics module (CER - dynamics ablation)",
      len(client2.calls) == 1 and picked_d2 == [])
check("skills still retrieved in that ablation", len(picked_s2) == 1)

empty_buf = {"dynamics": {}, "skills": {}}
client3 = FakeClient()
check("cold buffer makes no LLM call", retrieve("g", "w", empty_buf, client3) == ([], []) and client3.calls == [])


print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
