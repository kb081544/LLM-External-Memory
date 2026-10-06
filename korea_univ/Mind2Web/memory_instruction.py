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
