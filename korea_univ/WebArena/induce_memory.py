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

import os
import re
import gzip
import pickle
import json
import random
import argparse
from functools import partial
import time

from prompts.memory_instruction import (
    SUCCESSFUL_SI, FAILED_SI, AWM_INSTRUCTION, AWM_EXAMPLE, ACE_REFLECTOR_SI, ACE_REFLECTOR_FI,
    REME_SUCCESS_SI, REME_FAILURE_SI, MEMP_BUILD_SI,
    CER_DYNAMICS_DISTILL_SI, CER_SKILLS_DISTILL_SI,
)
from utils.clients import CLIENT_DICT


def _extract_think_from_output(ai: dict) -> str:
    """Extract think text from agent info, falling back to chat_messages."""
    think = ai.get("think", "")
    if think:
        return think
    # Model may put reasoning outside <think> tags — extract text before <action>
    msgs = ai.get("chat_messages", [])
    if len(msgs) >= 3:
        output = str(msgs[2])
        action_idx = output.find("<action>")
        if action_idx > 0:
            raw = output[:action_idx].strip()
            raw = re.sub(r"</?think>", "", raw).strip()
            if raw:
                return raw
    return think


def extract_think_and_action(folder: str) -> tuple[list[str], list[str]]:
    """Extract think/action pairs from step pkl files."""
    step_files = sorted(
        [f for f in os.listdir(folder) if re.match(r"step_\d+\.pkl\.gz", f)],
        key=lambda f: int(re.findall(r"\d+", f)[0])
    )
    think_list = []
    action_list = []
    for f in step_files:
        try:
            with gzip.open(os.path.join(folder, f), 'rb') as fh:
                data = pickle.load(fh)
            ai = data.agent_info
            think = _extract_think_from_output(ai)
            action = ai.get("action", "")
            if not action:  # skip empty action steps
                continue
            think_list.append(think)
            action_list.append(action)
        except Exception:
            continue
    return think_list, action_list

def extract_cer_steps(folder: str, max_page_chars: int = 1500) -> list[dict]:
    """Like extract_think_and_action, but also carries each step's URL and a bounded slice of
    its accessibility tree. CER's dynamics module summarizes *pages* and pairs them with the
    URL that reaches them (paper 3.1), so unlike every other arm it has to see the observation
    and not just think/action. axtree_txt runs to tens of thousands of characters, so it is
    truncated per step -- a page summary needs the top of the tree (title, nav, main landmarks),
    not the whole DOM."""
    step_files = sorted(
        [f for f in os.listdir(folder) if re.match(r"step_\d+\.pkl\.gz", f)],
        key=lambda f: int(re.findall(r"\d+", f)[0])
    )
    steps = []
    for f in step_files:
        try:
            with gzip.open(os.path.join(folder, f), 'rb') as fh:
                data = pickle.load(fh)
            ai = data.agent_info
            action = ai.get("action", "")
            if not action:  # same skip rule as extract_think_and_action
                continue
            obs = data.obs if isinstance(data.obs, dict) else {}
            page = (obs.get("axtree_txt") or "")[:max_page_chars]
            steps.append({
                "url": obs.get("url", ""),
                "page": page,
                "think": _extract_think_from_output(ai),
                "action": action,
            })
        except Exception:
            continue
    return steps


def format_cer_trajectory(steps: list[dict]) -> str:
    """State-action trajectory in the shape CER's distillation prompts describe: each step
    shows where the agent was, what it saw, what it thought and what it did."""
    out = []
    for i, s in enumerate(steps, 1):
        out.append(
            f"Step {i}\n"
            f"URL: {s['url']}\n"
            f"Observation (truncated):\n{s['page']}\n"
            f"<think>\n{s['think']}\n</think>\n"
            f"<action>\n{s['action']}\n</action>"
        )
    return "\n\n".join(out)


def format_trajectory(think_list: list[str], action_list: list[list[str]]) -> str:
    trajectory = []
    for t, a in zip(think_list, action_list):
        # acts = '\n'.join(a)
        acts = a
        trajectory.append(f"<think>\n{t}\n</think>\n<action>\n{acts}\n</action>")
    return '\n\n'.join(trajectory)

def random_group_sample(d: dict, n) -> list:
    """Randomly sample n groups from the dictionary."""
    return [ex for v in d.values() for ex in random.sample(v, min(n, len(v)))]


def format_examples(examples: list[dict], flag=False) -> str:
    """Format examples to the prompt."""
    formatted_examples = []
    for ex in examples:
        trajectory = format_trajectory(ex["think_list"], ex["action_list"])
        formatted_examples.append(f"Query: {ex['query']}\nTrajectory:\n{trajectory}")
    # return '\n\n'.join(["## Concrete Examples"] + formatted_examples + ["## Summary Workflow"])
    if flag:
        return '\n\n'.join(["## Query and Trajectory Generated Using Previous Memory"] + formatted_examples + ["## Correctness Signal"]+ ["The result is CORRECT."] + ["## Updated Memory"])
    else:
        return '\n\n'.join(["## Query and Trajectory Generated Using Previous Memory"] + formatted_examples + ["## Correctness Signal"]+ ["The result is INCORRECT."] + ["## Updated Memory"])


def get_info(f: str, status: str = None) -> dict:
        
    # get query -> task objective
    task_id = os.path.basename(f).split("_")[0].split(".")[1]
    config_path = os.path.join("config_files", f"{task_id}.json")
    config = json.load(open(config_path))
    query = config["intent"]

    template_id = config["intent_template_id"]  # for deduplication

    # parse trajectory from step pkl files
    think_list, action_list = extract_think_and_action(f)

    # add to template dict
    sites = ", ".join(config.get("sites", []))
    if status == 'success':
        wdict = {"query": query, "template_id": template_id, "think_list": think_list, "action_list": action_list, "status": "success", "sites": sites}
    elif status == 'fail':
        wdict = {"query": query, "template_id": template_id, "think_list": think_list, "action_list": action_list, "status": "fail", "sites": sites}

    return wdict

def _strip_fence(text: str) -> str:
    """Models sometimes wrap a whole response in a code fence despite the prompt's format
    block not showing one -- strip it so the per-block regexes in reme/extract.py and
    memp/procedure.py aren't thrown off (same fix the ace branch applies inline)."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text


def main():
    # collect result directories, e.g., ["results/webarena.0", ...]
    args.result_dir = args.result_dir.split()

    cur_task = os.path.join(args.result_dir[0], args.task)

    # correctness signals for trajectories
    if args.criteria == "gt":
        reward = json.load(open(os.path.join(cur_task, "summary_info.json")))["cum_reward"]
    elif args.criteria == "autoeval":
        reward = json.load(open(os.path.join(cur_task, f"{args.model}_autoeval.json")))[0]["rm"]
    else:
        raise ValueError(f"Invalid criteria: {args.criteria}.")

    if reward == 1:
        status = "success"
    else:
        status = "fail"

    ex = get_info(cur_task, status)

    # Define the LLM client based on the model choice
    llm_client = CLIENT_DICT[args.model](model_name=args.model)

    # memory extraction based on the trajectory and user queries
    trajectory = format_trajectory(ex["think_list"], ex["action_list"])
    trajectory = f"**Query:** {ex['query']}\n\n**Trajectory:**\n{trajectory}"

    # Load autoeval thoughts if available, and append to trajectory so the
    # memory LLM knows why the task succeeded or failed.
    autoeval_thoughts = ""
    if args.criteria == "autoeval":
        autoeval_path = os.path.join(cur_task, f"{args.model}_autoeval.json")
        try:
            autoeval_data = json.load(open(autoeval_path))
            if isinstance(autoeval_data, list) and autoeval_data:
                autoeval_thoughts = autoeval_data[0].get("thoughts", "")
            elif isinstance(autoeval_data, dict):
                autoeval_thoughts = autoeval_data.get("thoughts", "")
        except Exception:
            pass

    extra: dict = {}  # per-mode fields added to the output record (cer writes two raw blocks)

    if args.memory_mode == "reasoningbank":
        if autoeval_thoughts:
            status_label = "succeeded" if ex['status'] == 'success' else "failed"
            trajectory += f"\n\nThe task {status_label} because: {autoeval_thoughts}"
        if ex['status'] == 'success':
            generated_memory_item, _ = llm_client.one_step_chat(trajectory, system_msg=SUCCESSFUL_SI, temperature=1.0)
        else:
            generated_memory_item, _ = llm_client.one_step_chat(trajectory, system_msg=FAILED_SI, temperature=1.0)

    elif args.memory_mode == "awm":
        if ex['status'] == 'success':
            generated_memory_item, _ = llm_client.one_step_chat(trajectory, system_msg=AWM_INSTRUCTION + AWM_EXAMPLE, temperature=0.7)

    elif args.memory_mode == "synapse":
        if ex['status'] == 'success':
            generated_memory_item = trajectory

    elif args.memory_mode == "ace":
        # ACE's Reflector step (see prompts/memory_instruction.py's ACE_REFLECTOR_SI/FI
        # docstring for why this is a reimplementation, not an import of ace-agent/ace).
        # Produces fresh candidate bullets only -- the playbook-aware
        # dedup/helpful-harmful-vote step happens afterward in ace/curate.py, which has
        # access to the persistent playbook state this stateless call doesn't.
        system_msg = ACE_REFLECTOR_SI if ex['status'] == 'success' else ACE_REFLECTOR_FI
        generated_memory_item, _ = llm_client.one_step_chat(trajectory, system_msg=system_msg, temperature=0.7)
        # Models sometimes wrap the whole response in a code fence despite the prompt's
        # example not showing one -- strip it so ace/curate.py's per-block "[section] ..."
        # regex isn't thrown off by a leading/trailing ``` line (same fix Mind2Web/memory.py
        # already applies to its own induce_memory() output).
        generated_memory_item = (generated_memory_item or "").strip()
        if generated_memory_item.startswith("```"):
            generated_memory_item = re.sub(r"^```[a-zA-Z]*\n?", "", generated_memory_item)
            generated_memory_item = re.sub(r"\n?```$", "", generated_memory_item)

    elif args.memory_mode == "reme":
        # ReMe's per-trajectory extraction (paper 3.2): success-pattern recognition on a
        # success, failure analysis on a failure. The third analysis (comparative) needs a
        # success/fail PAIR and runs afterwards in pipeline_memory.py via reme/compare.py,
        # for the same reason ACE's curation step lives outside this stateless call.
        system_msg = REME_SUCCESS_SI if ex['status'] == 'success' else REME_FAILURE_SI
        generated_memory_item, _ = llm_client.one_step_chat(trajectory, system_msg=system_msg, temperature=0.7)
        generated_memory_item = _strip_fence(generated_memory_item)

    elif args.memory_mode == "memp":
        # Memp's BUILD step (paper 4.2): one generalized procedure per trajectory. Whether it
        # is actually appended, skipped, or used to revise an existing procedure in place is
        # the UPDATE step (memp/update.py), applied in pipeline_memory.py -- the paper treats
        # build and update as separate axes, so they stay separate here.
        generated_memory_item, _ = llm_client.one_step_chat(trajectory, system_msg=MEMP_BUILD_SI, temperature=0.7)
        generated_memory_item = _strip_fence(generated_memory_item)

    elif args.memory_mode == "cer":
        # CER's two distillation modules (paper 3.1): dynamics (pages + URLs) and skills
        # (sub-goal procedures), each its own call with its own prompt. Both run on successes
        # *and* failures -- the paper's main setting distills from both, and filtering to
        # successes is the separate CER_success ablation (5.6).
        # The buffer's current contents go into each prompt so the model can answer
        # "Summarized before" instead of re-distilling what is already stored (its own dedup
        # mechanism, prompt notes in Fig. 3/4). Parsing and merging happen in
        # pipeline_memory.py via cer/distill.py, which owns the persistent buffer -- same
        # split as ace/reme/memp above.
        from cer.buffer import load_buffer, format_existing_dynamics, format_existing_skills

        buf = load_buffer(os.path.join(args.cer_dir, "buffer.json")) if args.cer_dir else {"dynamics": {}, "skills": {}}
        cer_steps = extract_cer_steps(cur_task, max_page_chars=args.cer_page_chars)
        cer_trajectory = format_cer_trajectory(cer_steps)
        website = ex.get("sites") or "unknown"

        dyn_prompt = (
            f"Overall goal of the trajectory: {ex['query']}\n"
            f"Current website: {website}\n"
            f"Existing summarized pages:\n{format_existing_dynamics(buf)}\n"
            f"Human user trajectory:\n{cer_trajectory}"
        )
        cer_dynamics_raw, _ = llm_client.one_step_chat(
            dyn_prompt, system_msg=CER_DYNAMICS_DISTILL_SI, temperature=0.1)

        skills_prompt = (
            f"Overall goal: {ex['query']}\n"
            f"Current website: {website}\n"
            f"Existing skills:\n{format_existing_skills(buf)}\n"
            f"Human user trajectory:\n{cer_trajectory}"
        )
        cer_skills_raw, _ = llm_client.one_step_chat(
            skills_prompt, system_msg=CER_SKILLS_DISTILL_SI, temperature=0.1)

        extra = {
            "cer_dynamics_raw": _strip_fence(cer_dynamics_raw or ""),
            "cer_skills_raw": _strip_fence(cer_skills_raw or ""),
        }
        generated_memory_item = ""

    # write memory to jsonl file
    record = {
        "task_id": args.task.split(".")[-1],
        "query": ex["query"],
        "think_list": ex["think_list"],
        "action_list": ex["action_list"],
        "status": ex["status"],
        "memory_items": generated_memory_item.split("\n\n"),
        "template_id": ex["template_id"]
    }
    record.update(extra)
    with open(args.output_path, 'a') as f:
        f.write(json.dumps(record) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="results_base_new",
                        help="Path to the result directory. Support multiple directories separated by space.")
    parser.add_argument("--output_path", type=str, default=None, required=True,
                        help="Path to the output file.")
    parser.add_argument("--criteria", type=str, default="autoeval", choices=["gt", "autoeval"])
    parser.add_argument("--model", type=str, default="gemini-2.5-flash",
                        choices=["gpt-3.5", "gpt-4", "gpt-4o", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-2.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "agy-claude-sonnet-4-6", "ccli-sonnet"])
    parser.add_argument("--task", type=str, default="webarena.47")
    parser.add_argument("--memory_mode", type=str, default="reasoningbank")
    parser.add_argument("--cer_dir", type=str, default=None,
                        help="cer only: buffer directory, so the distillation prompts can be "
                             "shown what is already stored and answer 'Summarized before'.")
    parser.add_argument("--cer_page_chars", type=int, default=1500,
                        help="cer only: per-step accessibility-tree slice given to the dynamics "
                             "module. CER is the one arm that must see page content, not just "
                             "think/action.")
    args = parser.parse_args()

    main()