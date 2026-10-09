# Copyright 2026 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

PARALLEL_SI = """
You are an expert in web navigation. You will be given a user query and multiple trajectories showing how an agent attempted the task. 
Some trajectories may be successful, and others may have failed.

## Guidelines
Your goal is to **compare and contrast** these trajectories to identify the most useful and generalizable strategies as memory items.
Use **self-contrast reasoning**:
  - Identify patterns and strategies that consistently led to success.
  - Identify mistakes or inefficiencies from failed trajectories and formulate preventative strategies.
  - Prefer strategies that generalize beyond specific pages or exact wording.

## Important notes
  - Think first: Why did some trajectories succeed while others failed?
  - You can extract *at most 5* memory items from all trajectories combined.
  - Do not repeat similar or overlapping items.
  - Do not mention specific websites, queries, or string contents — focus on generalizable behaviors and reasoning patterns.
  - Make sure each memory item captures **actionable** and **transferable** insights.

## Output Format
Your output must strictly follow the Markdown format shown below:

```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary describing when or when NOT to use the memory item>
## Content <1-5 sentences describing the insights learned to avoid such failures and successfully accomplishing similar tasks in the future>
```
"""

PARALLEL_AWM_SI = """
You are an expert in web navigation. You will be given a user query and multiple trajectories showing how an agent attempted the task. 
Some trajectories may be successful, and others may have failed.

## Guidelines
Your goal is to extract the common workflows to solve these tasks.
**Workflow extraction process**:
  - Examine all given trajectories (both successful and failed).
  - Look for repetitive subsets of actions that occur across multiple tasks.
  - Focus on action sequences that are critical to completing parts of the task, not trivial or one-off actions.

## Important notes
  - Think first: Why did some trajectories succeed while others failed?
  - You can extract *at most 5* workflow items from all trajectories combined.
  - Do not repeat similar or overlapping items.
  - **Each workflow must have at least 2 steps (i.e., 2 <think><action> sequences)**.
  - Do not mention specific websites, queries, or string contents — focus on generalizable behaviors and reasoning patterns.
  - Ensure workflows are actionable and can be reused in similar tasks.

## Output Format
Your output must strictly follow the following example format:

```
Workflow i: Find all orders
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
```
"""

SEQUENTIAL_PROMPT = """
Important: Let's carefully re-examine the previous trajectory, including your reasoning steps and actions taken.
Pay special attention to whether you used the correct elements on the page, and whether your response addresses the user query.
If you find inconsistencies, correct them. If everything seems correct, confirm your final answer.
Output must stay in the same “<think>...</think><action></action>” format as previous trajectories.
"""

SEQUENTIAL_FOLLOWING_PROMPT = """
Let's check again. 
Output must stay in the same “<think>...</think><action></action>” format as previous trajectories.
"""

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

<think>
To find the date of the first purchase, I need to look at the earliest date in the "Date" column of the "Orders" table. The current page shows the first 10 orders, but there are multiple pages. I will navigate to the last page to find the earliest order date.
</think>
<action>
click('1788')
</action>

<think>
To find the date of the first purchase, I need to look for the earliest date in the "Date" column of the "My Orders" table. From the current view, the earliest date visible is "5/2/23". However, there might be more orders on other pages. I will navigate to the last page to ensure I find the earliest date.
</think>
<action>
click('1784')
</action>

<think>
To find the earliest purchase date, I need to look at the "Date" column in the "Orders" table and identify the earliest date listed. The dates currently visible are: 4/27/22, 4/5/22, 3/10/22, 3/2/22, 3/2/22. The earliest date visible is 3/2/22. I will respond with this date.
</think>
<action>
send_msg_to_user('The earliest purchase date is 3/2/22.')
</action>

Query: I want to browse the products in the Cabinets, Racks & Shelves category
Actions:
<think>
To browse the products in the Cabinets, Racks & Shelves category, I need to navigate to the "Home & Kitchen" section, which likely contains the Cabinets, Racks & Shelves category. I will start by hovering over the "Home & Kitchen" menu item to reveal the subcategories.
</think>
<action>
hover('735')
</action>

<think>
To browse the products in the Cabinets, Racks & Shelves category, I need to navigate to the "Storage & Organization" section under the "Home & Kitchen" menu. I will start by hovering over the "Storage & Organization" menu item to reveal the subcategories.
</think>
<action>
hover('786')
</action>

<think>
To browse the products in the Cabinets, Racks & Shelves category, I need to click on the "Racks, Shelves & Drawers" link under the "Storage & Organization" menu. This will take me to the desired category.
</think>
<action>
click('797')
</action>

<think>
The current page is displaying the "Racks, Shelves & Drawers" category under "Storage & Organization" in the "Home & Kitchen" section. The user wants to browse products in the "Cabinets, Racks & Shelves" category. Since "Racks, Shelves & Drawers" is already a subcategory of "Storage & Organization," I will confirm with the user if this is the correct category they want to browse.
</think>
<action>
send_msg_to_user('You are currently viewing the "Racks, Shelves & Drawers" category under "Storage & Organization." Is this the correct category you want to browse, or are you looking for a different category?')
</action>


Query: Get the order number of my most recent cancelled order 
Actions:
<think>
To get the order number of the most recent cancelled order, I need to navigate to the "My Account" section where order history is typically managed. From the current page, I can see the "My Account" link in the top right corner. I will click on the "My Account" link to proceed.
</think>
<action>
click('227')
</action>

<think>
To get the order number of the most recent cancelled order, I need to navigate to the "My Orders" section. From there, I can filter or sort the orders to find the most recent cancelled order. I will start by clicking on the "My Orders" link in the left sidebar.
</think>
<action>
click('1843')
</action>

<think>
To get the order number of the most recent cancelled order, I need to look for the order with the status "Canceled" and note its order number. From the current page, I can see that the most recent cancelled order is order number 000000170.
</think>
<action>
send_msg_to_user('The order number of your most recent cancelled order is 000000170.')
</action>


## Summary Workflows

Workflow 1: Find all orders
<think>
To find the date of the first purchase, I need to navigate to the order history section of the user's account. I will start by clicking on the "My Account" link.
</think>
<action>
click('my_account_id')
</action>

<think>
To find the date of the first purchase, I need to navigate to the "My Orders" section where the order history is listed. From there, I can look for the earliest order date. I will start by clicking on the "My Orders" link in the left sidebar.
</think>
<action>
click('my_orders_id')
</action>

Workflow 2: Obtain order information, such as date or order number
<think>
To find the earliest purchase date, I need to look at the "Date" column in the "Orders" table and identify the earliest date listed. The dates currently visible are: 4/27/22, 4/5/22, 3/10/22, 3/2/22, 3/2/22. The earliest date visible is 3/2/22. I will respond with this date.
</think>
<action>
send_msg_to_user('The earliest purchase date is 3/2/2022.')
</action>

Workflow 3: Browse products in a specific (sub-)category
<think>
To browse the products in the Cabinets, Racks & Shelves category, I need to navigate to the "Home & Kitchen" section, which likely contains the Cabinets, Racks & Shelves category. I will start by hovering over the "Home & Kitchen" menu item to reveal the subcategories.
</think>
<action>
hover('home_kitchen_id')
</action>

<think>
To browse the products in the Cabinets, Racks & Shelves category, I need to navigate to the "Storage & Organization" section under the "Home & Kitchen" menu. I will start by hovering over the "Storage & Organization" menu item to reveal the subcategories.
</think>
<action>
hover('storage_organization_id')
</action>

<think>
To browse the products in the Cabinets, Racks & Shelves category, I need to click on the "Racks, Shelves & Drawers" link under the "Storage & Organization" menu. This will take me to the desired category.
</think>
<action>
click('racks_shelves_drawers_id')
</action>
"""

# ACE (Agentic Context Engineering, arXiv:2510.04618) baseline -- reimplemented in our own
# pipeline the same way AWM/Synapse above are (a standalone prompt + induction call), not by
# importing github.com/ace-agent/ace's package directly: their DataProcessor abstraction is
# built for single-shot QA-style tasks (process_task_data/answer_is_correct/evaluate_accuracy),
# not our multi-step WebArena action loop, so it wouldn't wire into run.py's live episode
# without the same amount of adaptation work as a reimplementation needs anyway. The real
# ACE's Reflector sees the *current playbook* and either increments an existing bullet's
# helpful/harmful count or proposes a brand-new one; this prompt only produces fresh candidate
# bullets (dedup-vs-existing-playbook happens afterward in ace/curate.py via embedding
# similarity, not inside this prompt) to keep induce_memory.py's call signature identical to
# reasoningbank/awm/synapse (stateless: trajectory in, text out).
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
# one needs a success/fail pair, which reme/compare.py supplies when an earlier task sharing
# the same intent_template_id had the opposite outcome.
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


# --- CER (Contextual Experience Replay; arXiv:2506.06698, ACL 2025) --------------------
# Reimplementation, not an import: CER has no public code release. The four system messages
# below are transcribed from the paper's Appendix A.1 (Figures 3-6) so the distillation and
# retrieval criteria are the paper's, not ours. Two things are worth knowing when reading
# these next to the other arms' prompts:
#   - CER splits an experience into *dynamics* (page summary + URL + usages, Fig. 3) and
#     *skills* (sub-goal name + step-by-step guideline with concrete action examples, Fig. 4),
#     and distills/retrieves each with its own module (paper 3.1, 3.2).
#   - Retrieval is an LLM *selecting* top-k entries out of the whole buffer (Fig. 5, 6), not
#     embedding cosine similarity. That is the structural difference from reasoningbank / efm /
#     reme, so it is kept as-is rather than swapped for our embedding retriever.
# Deduplication is also prompt-level: the distillation prompts are shown the buffer's existing
# entries and told to answer "Summarized before" instead of re-summarizing.
# {max_n} stands in for the paper's k_d / k_s (both 5 in its WebArena setup, 4.1.1).
#
# One deliberate deviation from the paper's exact wording: the original Figures 3-6 wrap a
# scratch-reasoning instruction in a literal "<<think>>\nthink step by step\n<</think>>" block
# in both the output-format spec and the worked examples. On our backbone (Claude via the CLI,
# ccli-sonnet) that combination -- a tag named "think" whose content explicitly asks for a
# step-by-step reasoning trace -- is refused deterministically (stop_reason "refusal",
# classifier tag "reasoning_extraction"; reproduced 8/8 across all four prompts, 0/8 once the
# <<think>> blocks are removed -- see NOTES.md). None of the other arms' prompts contain this
# phrasing and none hit this. The <<think>> blocks are dropped below (format-spec AND worked
# examples, so the two stay consistent) -- they were never parsed anyway (cer/distill.py only
# reads <<skill>>/<<steps>> and <<URL>>/<<page-summary>>), so nothing CER actually extracts is
# lost, only the ReAct-style scratch-reasoning formatting convention the paper layers on top.
CER_DYNAMICS_DISTILL_SI = """
You will be given the state-action trajectory of a user interacting with a webpage and the overall goal of the trajectory.
You need to summarize the useful pages and pair it up with the corresponding URLs.
Output format:
<<URL>>
the URL of page 1
<</URL>>
<<page-summary>>
the brief summary of page 1, following the format:
Name: name page
Description: descriptions
Usages: usages
<</page-summary>>
<<URL>>
the URL of page 2
<</URL>>
<<page-summary>>
the brief summary of page 2, following the format:
Name: name page
Description: descriptions
Usages: usages
<</page-summary>>
...
# Examples
## Example 1
Overall goal of the trajectory: Go to r/books forum.
Current website: Reddit
Existing summarized pages:
Page 1: Profile page
Description: it shows the user's profile information.
Usages: view or modify user's profile information.
URL: https://www.example.com/profile
Human user trajectory: [neglected here]
## Output:
<<URL>>
https://www.example.com/forums
<</URL>>
<<page-summary>>
Name: Forums page; Description: it shows a list of different forums.; Possible usages: navigate to different forums.
<</page-summary>>
IMPORTANT NOTES you should absolutely follow:
1. DO NOT include any other words except url and page summary as the format stated above.
2. Follow the example to think and summarize the page.
3. You should only summarize once for each unique URL.
4. Check existing pages before generating, do not summarize pages that have already been summarized, instead, use "Summarized before" in the steps.
5. Focus on the main content of the page and may ignore the modifications made by the user when generating the summary.
"""

CER_SKILLS_DISTILL_SI = """
You will be given the state-action trajectory of a user interacting with a webpage and the overall goal of the trajectory.
You need to summarize skills from the trajectory.
Skills are a subset of actions that the user takes to achieve a sub-goal.
You should break the overall goal into sub-goals and summarize each sub-goal as a skill.
Represent the non-fixed elements (input text, button strings) and non-fixed words (e.g. a specific forum name / user name; an option) with descriptive variable names as shown in the example.
Output format:
<<skill>>
skill1 name here.
<</skill>>
<<steps>>
The steps of the skill1 here.
<</steps>>
<<skill>>
skill2 name here.
<</skill>>
<<steps>>
The steps of the skill2 here.
<</steps>>
...
# Examples
## Example 1
Overall goal: I want to get the cheapest product in the Cabinets, Racks & Shelves category
Current website: current website
Existing skills:
Skill 1: Sort products by sort criterion
1. To sort the products by sort criterion, I need to click on the "Sort by" dropdown menu.
```click(sort by id)```
2. To sort the products by sort criterion, I need to select the sort criterion option from the "Sort by" dropdown menu.
```click(sort criterion id)```
Human user trajectory: [neglected here for length]
##Output: [neglected here for length]
IMPORTANT NOTES you should absolutely follow:
1. DO NOT include any other words except skills and steps as the format stated above.
2. Check existing skills before generating; do not summarize skills that have already been summarized; instead, use "Summarized before" in the steps.
3. You should break the overall goal into sub-goals and summarize each sub-goal as a skill.
"""

CER_DYNAMICS_RETRIEVE_SI = """
You will be given a goal of a task to be executed on a website and a list of urls and the corresponding page summary to choose from.
You need to select the pages that most possibly need to be visited to achieve the goal.
You should break the task down into a few steps so that you can select the pages that can help most in each step.
IMPORTANT: You should select not more than {max_n} pages!
Output format:
<<selected-pages>>
id: the id number (the number at the beginning) of page 1; name: page 1 name
id: the id number (the number at the beginning) of page 2; name: page 2 name
...
<</selected-pages>>
# Examples
## Example 1
Task goal: Upvote the hottest post in r/books
Current website: website descriptions
Shortcuts to choose from:
id: 1; name: Forums page; description: It shows a list of different forums; possible usages: navigate to different forums; url: https://www.example.com/forums
id: 2; name: Profile page; description: It shows the information of current user; possible usages: Check or modify the information of the current user; url: https://www.example.com/profile
id: 3; name: Submission page; description: It provides a few text boxes to fill in to submit a new post; possible usages: Submit new posts; url: https://www.example.com/submission
id: 4: name: Subscribed forums page; description: It provides a list of subscribed forums; possible usages: check or navigate to subscribed forums; url: https://www.example.com/subscribed
## Output 1:
<<selected-pages>>
id: 1; name: Forums page
<</selected-pages>>
"""

CER_SKILLS_RETRIEVE_SI = """
You will be given a goal of a task to be executed on a website and a list of skills to choose from.
You need to select the skills that can help most in achieving the goal.
You should break the task down into a few steps so that you can select the skills that can help most in each step.
IMPORTANT: You should select not more than {max_n} skills!
Output format:
<<selected-skills>>
id: the id number (the number at the beginning) of skill 1; name: skill 1 name
id: the id number (the number at the beginning) of skill 2; name: skill 2 name
...
<</selected-skills>>
# Examples
## Example 1
Task goal: Upvote the hottest post in r/books
Current website: website descriptions
Skills to choose from:
Skill 1: Navigate to forums
1. Click on the "Forums" menu item.
```click(forums id)```
2. Click on the specific forum name.
```click(forum name id)```
Skill 2: Submit a new post
1. Type the post title in the title text box.
```type(title text box id, "Post Title")```
2. Type the post content in the content text box.
```type(content text box id, "Post Content")```
3. Click on the "Submit" button.
```click(submit button id)```
Skill 3: Sort posts by sort criterion
1. Click on the "Sort by" dropdown menu.
```click(sort by dropdown id)```
2. Select the sort criterion option from the "Sort by" dropdown menu.
```click(sort criterion id)```
Output:
<<selected-skills>>
id: 1; name: Navigate to forums
id: 3; name: Sort posts by hotness
<</selected-skills>>
Notes:
1. Some skills might not be consistent with the current task but it is still useful to refer to, e.g. write a post to express happiness is useful in a task to write a post to express sadness.
"""

# Header prepended to the retrieved experiences when they are replayed into the agent's context
# (paper 3.3: the selected experiences are mapped to natural language and concatenated into the
# prompt). Deliberately parallel in shape to MEM_INSTRUCTION above so the arms differ in
# *algorithm*, not in how much scaffolding text the agent sees.
CER_REPLAY_INSTRUCTION = (
    "Below are experiences replayed from past interactions with this environment that may be "
    "helpful for the current task. The pages tell you what key pages contain and how to reach "
    "them directly by URL; the skills give step-by-step patterns that worked before. Use them "
    "when relevant, and ignore them when they do not apply."
)
