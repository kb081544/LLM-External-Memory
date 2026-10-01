"""MaTTS parallel scaling for WebArena (ReasoningBank paper Section 3.3, Appendix A.2
Figure 12): run k independent trajectories for the same task against k separate live
site instances, then use an LLM Best-of-N judge (the paper's own prompt, verbatim) to
pick the single best one. Only that selected trajectory is scored (autoeval) and used
for memory induction.

Unlike Mind2Web (offline, stateless replay), WebArena sites carry live state (cart,
order history, ...), so "k trajectories for the same task" only means something if each
trajectory runs against its own untouched site instance -- see pipeline_matts.py, which
fans out to k Docker containers on different ports rather than reusing one shared site.
"""

from __future__ import annotations

import json
import re

from induce_memory import extract_think_and_action, format_trajectory

BON_SYSTEM = """You are an expert in evaluating web navigation agent trajectories. You will be given the user query, and {N} candidate trajectories, each representing a sequence of steps for solving the same task. Your job is to select the single best trajectory that most effectively and efficiently solves the task, and explain your reasoning.

## Input Format:
Each trajectory consists of multiple steps. For each step, you will be provided:
-step_num: Step index in the trajectory.
-action_output: The action the agent takes (click, type, scroll, etc.).
-think_output: The agent's reasoning or plan before taking the action.

## Evaluation Criteria:

### Progress Toward Goal
1. How well the trajectory advances toward completing the user's task.
2. Reward tangible, meaningful progress; penalize minimal or no advancement.
3. Consider both individual step contributions and overall progress.

### Trajectory Efficiency
1. How efficiently the trajectory achieves progress given the number and complexity of steps.
2. Reward significant progress in fewer steps.
3. Favor better value-to-depth ratios.
4. Reward efficient search space exploration.

### Loop Detection
Identify loops or redundant actions.
1. Real Loops: Repeating identical observations and actions with no added value.
2. Benign Repetitions: Slight variations that still yield new information.
3. Penalize real loops heavily; penalize benign repetitions only if they waste effort.

### Error Severity and Stability
Assess severity of errors:
1. Fatal/Blocking: Major penalty.
2. Significant: Moderate penalty.
3. Minor/Recoverable: Minor penalty.
4. Penalize unstable or incoherent model reasoning.
5. Consider whether errors prevent goal completion.

### Overall Trajectory Quality
1. Logical flow of steps, clarity of strategy, and coherence.
2. Balanced exploration vs. exploitation.
3. Closeness to final goal.
4. Reward consistent progress and coherent planning.

## Output Format:
Return the evaluation as a JSON object:
```
{{ "index": [best_trajectory_index], "analysis": "Detailed reasoning explaining why this trajectory is the best, referencing progress, efficiency, loop detection, error severity, and overall quality." }}
```
"""


def format_trajectories_from_dirs(task_dirs: list[str]) -> list[str]:
    """Extracts think/action pairs from each trajectory's step pkl files and formats
    them the same way induce_memory.py does (<think>...</think><action>...</action>
    per step)."""
    formatted = []
    for folder in task_dirs:
        think_list, action_list = extract_think_and_action(folder)
        formatted.append(format_trajectory(think_list, action_list))
    return formatted


def select_best_trajectory(client, query: str, task_dirs: list[str]) -> tuple[int, str | None]:
    """task_dirs: the k result folders for one task (one per parallel trial).
    Returns (0-based index of the selected trajectory, raw judge response or None)."""
    if len(task_dirs) <= 1:
        return 0, None

    trajectories = format_trajectories_from_dirs(task_dirs)
    system_msg = BON_SYSTEM.format(N=len(trajectories))
    traj_blocks = "\n".join(f"Trajectory {i}: {t}" for i, t in enumerate(trajectories, 1))
    prompt = f"Query: {query}\n{traj_blocks}"

    try:
        raw_text, _ = client.one_step_chat(prompt, system_msg=system_msg)
    except Exception as exc:  # noqa: BLE001 - a BoN judge hiccup shouldn't kill the whole run
        return 0, f"BoN judge error: {type(exc).__name__}: {exc}"

    raw_text = raw_text or ""
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            idx_raw = obj["index"]
            idx = int(idx_raw[0] if isinstance(idx_raw, list) else idx_raw) - 1
            if 0 <= idx < len(task_dirs):
                return idx, raw_text
        except (json.JSONDecodeError, TypeError, ValueError, KeyError, IndexError):
            pass
    return 0, raw_text
