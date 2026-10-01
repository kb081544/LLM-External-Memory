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
