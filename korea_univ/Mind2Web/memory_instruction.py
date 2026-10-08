"""ReasoningBank induction prompts, copied verbatim from
WebArena/prompts/memory_instruction.py (SUCCESSFUL_SI / FAILED_SI only -- Mind2Web only
needs the reasoningbank mode, not AWM/Synapse/MaTTS variants). Kept as a local copy so
Mind2Web/ stays self-contained, matching the utils/clients.py copy in this same folder.
"""

SUCCESSFUL_SI = """
You are an expert in web navigation. You will be given a user query, the corresponding trajectory that represents **how an agent successfully accomplished the task**.

## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's successful trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.

## Important notes
  - You must first think why the trajectory is successful, and then summarize the insights.
  - You can extract *at most 3* memory items from the trajectory.
  - You must not repeat similar or overlapping items.
  - Prefer concrete, actionable procedures over abstract principles. Do not embed specific product names, queries, or literal string contents from the task.

## Output Format
Your output must strictly follow the Markdown format shown below:

```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary describing when or when NOT to use the memory item>
## Content <1-3 sentences describing the insights learned to successfully accomplishing similar tasks in the future>
```
"""

FAILED_SI = """
You are an expert in web navigation. You will be given a user query, the corresponding trajectory that represents **how an agent attempted to resolve the task but failed**.

## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's failed trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.

## Important notes
  - You must first reflect and think why the trajectory failed, and then summarize what lessons you have learned or strategies to prevent the failure in the future.
  - You can extract *at most 3* memory items from the trajectory.
  - You must not repeat similar or overlapping items.
  - Prefer concrete, actionable recovery procedures over abstract principles. Do not embed specific product names, queries, or literal string contents from the task.

## Output Format
Your output must strictly follow the Markdown format shown below:

```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary describing when or when NOT to use the memory item>
## Content <1-3 sentences describing the insights learned to avoid such failures and successfully accomplishing similar tasks in the future>
```
"""

MEM_INSTRUCTION = (
    "Below are some memory items that I accumulated from past interaction from "
    "the environment that may be helpful to solve the task. You can use it when "
    "you feel it's relevant. In each step, please first explicitly discuss if you "
    "want to use each memory item or not, and then take action."
)

# AWM (Agent Workflow Memory baseline) and ACE (Agentic Context Engineering, see
# WebArena/prompts/memory_instruction.py's ACE_REFLECTOR_SI/FI docstring for why this is a
# reimplementation, not an import of ace-agent/ace) -- copied verbatim from
# WebArena/prompts/memory_instruction.py, same reasoning as the SUCCESSFUL_SI/FAILED_SI copy
# above. AWM_EXAMPLE's click('227')-style content is illustrating *output format* only, not
# literal Mind2Web task content, so it doesn't need to be benchmark-specific.
AWM_INSTRUCTION = """
Given a list of web navigation tasks, your task is to extract the common workflows to solve these tasks.
Each given task contains a natural language instruction, and a series of actions to solve the task. You need to find the repetitive subset of actions across multiple tasks, and extract each of them out as a workflow.
Each workflow should be a commonly-reused sub-routine of the tasks. Do not generate similar or overlapping workflows. Each workflow should have at least two steps. Represent the non-fixed elements (input text, button strings) with descriptive variable names as shown in the example.
Keep the values of invariant elements, e.g., id of "Search" or "Customers", as they will share and stay invariant across tasks.
Try to generate as many workflows that can cover all the tasks in the input list.
"""

AWM_EXAMPLE = """
## Concrete Examples

Query: What is the date when I made my first purchase on this site?
Actions:
<think>
To find the date of the first purchase, I need to navigate to the order history section of the user's account. I will start by clicking on the "My Account" link.
</think>
<action>
click('227')
</action>

<think>
To find the date of the first purchase, I need to navigate to the "My Orders" section where the order history is listed. From there, I can look for the earliest order date. I will start by clicking on the "My Orders" link in the left sidebar.
</think>
<action>
click('1843')
</action>
"""

ACE_REFLECTOR_SI = """
You are an expert in web navigation, acting as the "Reflector" in an Agentic Context Engineering
(ACE) pipeline. You will be given a user query and the trajectory of an agent that
**successfully accomplished the task**.

## Guidelines
Extract concrete, reusable insight bullets a "playbook" of strategies could add. Each bullet
should be a single, standalone, actionable tip -- not a narrative summary of this one task.

## Important notes
  - You can extract *at most 3* bullets.
  - Do not repeat similar or overlapping bullets.
  - Do not embed specific product names, queries, or literal string contents from the task.
  - Tag each bullet with ONE section: "strategy" (a reusable procedure), "formula" (a
    reusable fact/calculation rule), or "mistake" (a pitfall worth naming even on a success,
    e.g. a near-miss the agent caught).

## Output Format
Your output must strictly follow this format, one bullet per block, separated by a blank line:

```
[strategy] <one self-contained, actionable sentence>

[formula] <one self-contained, actionable sentence>
```
"""

ACE_REFLECTOR_FI = """
You are an expert in web navigation, acting as the "Reflector" in an Agentic Context Engineering
(ACE) pipeline. You will be given a user query and the trajectory of an agent that
**attempted the task but failed**.

## Guidelines
Extract concrete, reusable insight bullets a "playbook" of strategies could add so this failure
mode is avoided next time. Each bullet should be a single, standalone, actionable tip -- not a
narrative summary of this one task.

## Important notes
  - You can extract *at most 3* bullets.
  - Do not repeat similar or overlapping bullets.
  - Do not embed specific product names, queries, or literal string contents from the task.
  - Tag each bullet with ONE section: "strategy" (a reusable procedure to use instead),
    "formula" (a reusable fact/calculation rule), or "mistake" (the specific pitfall to avoid).

## Output Format
Your output must strictly follow this format, one bullet per block, separated by a blank line:

```
[mistake] <one self-contained, actionable sentence>

[strategy] <one self-contained, actionable sentence>
```
"""


# --- ReMe (Remember Me, Refine Me; Findings of ACL 2026) -------------------------------
# Reimplemented here rather than imported from agentscope-ai/ReMe for the same reason ACE is
# (above): the released code is built around BFCL-V3 / AppWorld tool-call episodes with its
# own agent loop and vector store. The paper's extraction step runs THREE analyses over a
# task's trajectories -- success-pattern, failure-analysis, and a comparative analysis that
# contrasts a successful against a failed trajectory of the same task (3.2). The first two are
# per-trajectory and map directly onto our one-trajectory-per-task pipeline; the comparative
# one needs a success/fail pair, which reme/compare.py supplies when an earlier task on the same
# website had the opposite outcome (Mind2Web rows carry no intent_template_id).
# The schema asked for below is the paper's E = <omega (when to use), e (content),
# kappa (keywords), c (confidence)>; tau (tools) is filled in from the trajectory, not asked
# for here.
REME_SUCCESS_SI = """
You are an expert in web navigation performing **success pattern recognition** for an
experience library. You will be given a user query and the trajectory of an agent that
**successfully accomplished the task**.

## Guidelines
Distill the reusable principle that made this work, and state explicitly WHEN a future agent
should reach for it. The "when to use" line is what a future agent searches on, so write it as
a description of the *situation*, not of this particular task.

## Important notes
  - You can extract *at most 3* experiences.
  - Do not repeat similar or overlapping experiences.
  - Do not embed specific product names, queries, or literal string contents from the task.
  - Keywords: 2-5 short terms. Confidence: high, medium, or low.

## Output Format
Your output must strictly follow this format, one experience per block, separated by a blank line:

# Experience 1
## When to use <one sentence describing the situation this applies to>
## Content <1-3 sentences: the reusable strategy>
## Keywords <comma-separated terms>
## Confidence <high|medium|low>
"""

REME_FAILURE_SI = """
You are an expert in web navigation performing **failure analysis** for an experience library.
You will be given a user query and the trajectory of an agent that **attempted the task but
failed**.

## Guidelines
Name the pitfall and the corrective behaviour, and state explicitly WHEN a future agent should
watch for it. The "when to use" line is what a future agent searches on, so write it as a
description of the *situation*, not of this particular task.

## Important notes
  - You can extract *at most 3* experiences.
  - Do not repeat similar or overlapping experiences.
  - Do not embed specific product names, queries, or literal string contents from the task.
  - Keywords: 2-5 short terms. Confidence: high, medium, or low.

## Output Format
Your output must strictly follow this format, one experience per block, separated by a blank line:

# Experience 1
## When to use <one sentence describing the situation this applies to>
## Content <1-3 sentences: the pitfall and what to do instead>
## Keywords <comma-separated terms>
## Confidence <high|medium|low>
"""

REME_COMPARATIVE_SI = """
You are an expert in web navigation performing **comparative analysis** for an experience
library. You will be given one user query solved SUCCESSFULLY and a second, closely related
query where the agent FAILED, each with its trajectory.

## Guidelines
Identify what the successful trajectory did differently -- the decisive divergence, not a
summary of either run. State explicitly WHEN a future agent should apply that difference.

## Important notes
  - You can extract *at most 2* experiences.
  - Only report differences that plausibly explain the outcome gap.
  - Do not embed specific product names, queries, or literal string contents from the tasks.
  - Keywords: 2-5 short terms. Confidence: high, medium, or low.

## Output Format
Your output must strictly follow this format, one experience per block, separated by a blank line:

# Experience 1
## When to use <one sentence describing the situation this applies to>
## Content <1-3 sentences: the decisive difference and how to act on it>
## Keywords <comma-separated terms>
## Confidence <high|medium|low>
"""


# --- Memp (Exploring Agent Procedural Memory; Findings of ACL 2026) ---------------------
# Reimplementation, not an import of Zjunlp/MemP: the released code targets ALFWorld and
# TravelPlanner episodes (its own env wrappers + ReAct loop). Ported here are the two axes the
# paper actually ablates and that our pipeline can express: memory BUILD (script /
# trajectory / proceduralization, 4.2) and memory UPDATE (vanilla / validation / adjustment,
# 4.3). The paper's Facts and AveFact retrieval keys are not ported -- see comparisons/README.md.
MEMP_BUILD_SI = """
You are an expert in web navigation building a **procedural memory** entry. You will be given a
user query and the trajectory of an agent that completed it.

## Guidelines
Write the generalized procedure this trajectory instantiates: the ordered steps a future agent
should follow for this class of task. Replace task-specific literals with placeholders such as
{item-name} or {category}.

## Important notes
  - One procedure only, 3-8 steps.
  - Steps must be executable web actions (navigate, search, filter, click, read, compare), not
    narration of this one run.
  - Keep it short enough to stay useful when injected verbatim into a prompt.

## Output Format
Your output must strictly follow this format:

# Procedure
## When to use <one sentence describing the class of task this applies to>
## Steps
1. <step>
2. <step>
"""

MEMP_ADJUST_SI = """
You are an expert in web navigation **revising a procedural memory** entry (Memp's
"Adjustment" update strategy). You will be given an existing procedure, a user query, and the
trajectory of an agent that followed that procedure and **failed**.

## Guidelines
Rewrite the procedure in place so the same failure would not recur. Keep what still works;
change only what the failed trajectory shows to be wrong or missing.

## Important notes
  - Return the FULL revised procedure, not a diff.
  - Keep the same format and the 3-8 step budget.
  - Do not embed specific product names, queries, or literal string contents from the task.

## Output Format
Your output must strictly follow this format:

# Procedure
## When to use <one sentence describing the class of task this applies to>
## Steps
1. <step>
2. <step>
"""
