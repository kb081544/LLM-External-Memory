"""MaTTS parallel scaling (ReasoningBank paper Section 3.3 / Appendix A.2 Figure 12),
adapted to Mind2Web: generate k independent trajectories per task at temperature 0.7
under the same retrieved memory, then use an LLM Best-of-N judge to pick the single
most consistent, well-reasoned one. Only that selected trajectory is scored and used
for memory induction -- mirrors the paper's "self-contrast" idea that comparing
multiple attempts yields a more reliable trajectory (and therefore more reliable
memory) than any single sample.

Note: the paper's own Table 2 (Mind2Web results) has no "+MATTS" row -- MaTTS is only
reported there for WebArena (Table 1) and SWE-Bench-Verified (Table 3). This module is
a from-scratch application of the same idea to Mind2Web, not a reproduction of a
published number.
"""

from __future__ import annotations

import json
import re

BON_SYSTEM = """You are an expert in evaluating web element-selection agent trajectories on the Mind2Web benchmark. You will be given a task and N candidate trajectories, each representing the same agent's independent attempt to complete every step of that task by picking one candidate element + operation at each step.

## Input Format
Each trajectory lists its steps. For each step you are given the agent's chosen candidate description, the operation, the value (if any), and its one-sentence reasoning.

## Evaluation Criteria
### Grounded reasoning: does each step's reasoning cite specifics from the candidate list and the task, rather than being generic or hedging?
### Consistency: do the steps build toward the task coherently, without self-contradiction across steps?
### Confidence: prefer trajectories that commit to a clear choice at each step over ones that waver between multiple candidates.

## Output Format
Return ONLY a JSON object and nothing else:
{"index": <1-based index of the best trajectory>, "analysis": "<one short sentence>"}
"""


def format_trajectory_for_bon(steps: list[dict]) -> str:
    lines = []
    for s in steps:
        pred = s["prediction"]
        lines.append(
            f"Step {s['step']}: candidate={pred['backend_node_id']!r} "
            f"op={pred['operation']!r} value={pred['value']!r} reason={s.get('reason', '')!r}"
        )
    return "\n".join(lines)


def select_best_trajectory(one_step_chat, confirmed_task: str, trajectories: list[list[dict]]) -> tuple[int, str | None]:
    """Returns (0-based index of the selected trajectory, raw judge response or None)."""
    if len(trajectories) <= 1:
        return 0, None

    blocks = [
        f"### Trajectory {i}\n{format_trajectory_for_bon(steps)}"
        for i, steps in enumerate(trajectories, 1)
    ]
    prompt = f"Task: {confirmed_task}\n\n" + "\n\n".join(blocks)

    try:
        raw_text, _ = one_step_chat(prompt, system_msg=BON_SYSTEM, temperature=0.0)
    except Exception as exc:  # noqa: BLE001 - a BoN judge hiccup shouldn't kill the whole run
        return 0, f"BoN judge error: {type(exc).__name__}: {exc}"
    raw_text = raw_text or ""
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            idx = int(obj["index"]) - 1
            if 0 <= idx < len(trajectories):
                return idx, raw_text
        except (json.JSONDecodeError, TypeError, ValueError, KeyError):
            pass
    return 0, raw_text
