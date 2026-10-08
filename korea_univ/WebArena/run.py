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
import shutil
import argparse
import json
import logging
from pathlib import Path
import sys
import multiprocessing
import time, shutil


try:
    from utils import gemini_retry  # noqa: F401 — patches google.genai with backoff
except ImportError:
    pass

for _k in ("SHOPPING", "SHOPPING_ADMIN", "REDDIT", "GITLAB", "WIKIPEDIA", "MAP", "HOMEPAGE"):
    if f"WA_{_k}" in os.environ:
        os.environ[_k] = os.environ[f"WA_{_k}"]

from memory_management import select_memory, log_retrieval_event
import webarena_patch  # noqa: F401 — applies MemEvol llm_fuzzy_match prompt

from browsergym.experiments import ExpArgs, EnvArgs

from agents.legacy.agent import GenericAgentArgs
from agents.legacy.dynamic_prompting import Flags
from agents.legacy.utils.chat_api import ChatModelArgs


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def ensure_file(path: str):
    """Ensure that the file and its parent directories exist."""
    if path is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        open(path, "w").close()


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiment with hyperparameters.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemini-2.5-flash",
        help="Model name for the chat model.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="openended",
        help="Name of the Browsergym task to run. If 'openended', you need to specify a 'start_url'",
    )
    parser.add_argument(
        "--start_url",
        type=str,
        default="https://www.google.com",
        help="Starting URL (only for the openended task).",
    )
    parser.add_argument(
        "--slow_mo", type=int, default=500, help="Slow motion delay for the playwright actions."
    )
    parser.add_argument(
        "--headless",
        type=str2bool,
        default=True,
        help="Run the experiment in headless mode (hides the browser windows).",
    )
    parser.add_argument(
        "--record_video",
        type=str2bool,
        default=False,
        help="Record a Playwright video for later trajectory review.",
    )
    parser.add_argument(
        "--demo_mode",
        type=str2bool,
        default=True,
        help="Add visual effects when the agents performs actions.",
    )
    parser.add_argument(
        "--use_html", type=str2bool, default=False, help="Use HTML in the agent's observation space."
    )
    parser.add_argument(
        "--use_ax_tree",
        type=str2bool,
        default=True,
        help="Use AX tree in the agent's observation space.",
    )
    parser.add_argument(
        "--use_screenshot",
        type=str2bool,
        default=False,
        help="Use screenshot in the agent's observation space.",
    )
    parser.add_argument(
        "--multi_actions", type=str2bool, default=True, help="Allow multi-actions in the agent."
    )
    parser.add_argument(
        "--action_space",
        type=str,
        default="bid",
        choices=["python", "bid", "coord", "bid+coord", "bid+nav", "coord+nav", "bid+coord+nav"],
        help="",
    )
    parser.add_argument(
        "--use_history",
        type=str2bool,
        default=True,
        help="Use history in the agent's observation space.",
    )
    parser.add_argument(
        "--use_thinking",
        type=str2bool,
        default=True,
        help="Use thinking in the agent (chain-of-thought prompting).",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=30,
        help="Maximum number of steps to take for each task.",
    )
    parser.add_argument(
        "--memory_path",
        type=str,
        default=None,
        help="Path to the memory file to load for the agent.",
    )
    parser.add_argument(
        "--usage_log_path",
        type=str,
        default=None,
        help="If set, append which memory bank entries were retrieved for this task to this "
             "JSONL file (used by reasoningbank_pruned to compute per-entry utility later). "
             "No effect on any other mode.",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="./results",
        help="Path to the directory where the results will be saved.",
    )
    parser.add_argument(
        "--efm_dir",
        type=str,
        default=None,
        help="If set, use EFM retrieval (efm/retrieve.py) instead of baseline reasoningbank "
             "retrieval; the injected text still lands in --memory_path exactly like baseline, "
             "so agent.py/dynamic_prompting.py need no changes. Directory holds items.json, "
             "experiences.json, and the query/description embedding caches.",
    )
    parser.add_argument("--efm_k_cand", type=int, default=5)
    parser.add_argument("--efm_eta", type=float, default=0.1)
    parser.add_argument("--efm_gamma", type=float, default=0.98)
    parser.add_argument("--efm_n_inject", type=int, default=1)
    parser.add_argument(
        "--ace_dir",
        type=str,
        default=None,
        help="If set, inject the current ACE playbook (ace/playbook.py) into --memory_path "
             "instead of letting the reasoningbank-style select_memory() branch below run. "
             "No per-query retrieval -- ACE injects the whole (size-capped) evolving playbook "
             "every task, matching the paper's design.",
    )
    parser.add_argument("--ace_max_bullets", type=int, default=20)
    parser.add_argument(
        "--reme_dir",
        type=str,
        default=None,
        help="If set, retrieve ReMe experiences (reme/retrieve.py: cosine over each "
             "experience's usage-scenario embedding) instead of the reasoningbank-style "
             "select_memory() branch below. Injected text still lands in --memory_path, so "
             "the agent side is byte-for-byte the same mechanism as every other arm.",
    )
    parser.add_argument("--reme_top_k", type=int, default=3)
    parser.add_argument(
        "--memp_dir",
        type=str,
        default=None,
        help="If set, retrieve Memp procedures (memp/retrieve.py: cosine over the source "
             "task query, the paper's Key=Query condition) instead of the select_memory() "
             "branch below.",
    )
    parser.add_argument("--memp_top_k", type=int, default=1)
    parser.add_argument("--memp_build", type=str, default="proceduralization",
                        choices=["script", "trajectory", "proceduralization"],
                        help="Which of Memp's BUILD conditions to inject (paper 4.2).")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="DEBUG-level logging to console + experiment.log, so per-step "
             "'Starting step'/'Agent chose action' lines are visible live (normally "
             "silent -- only INFO+ is shown by default).",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    if args.ace_dir:
        ensure_file(args.memory_path)
        os.makedirs(args.ace_dir, exist_ok=True)  # first task of a fresh run: induce_memory.py's
        # --output_path {ace_dir}/_raw.jsonl (step 3) needs this dir to already exist.
        from ace.playbook import load_playbook, format_playbook_for_prompt

        playbook = load_playbook(f"{args.ace_dir}/playbook.json")
        block = format_playbook_for_prompt(playbook, max_bullets=args.ace_max_bullets)
        with open(args.memory_path, "w") as f:
            f.write(block + ("\n" if block else ""))

    elif args.efm_dir:
        ensure_file(args.memory_path)
        os.makedirs(args.efm_dir, exist_ok=True)  # first task of a fresh run: nothing else
        # creates this dir before store.load_cache()'s create-empty-cache-file fallback runs.

        from efm import store
        from efm.retrieve import retrieve
        from efm.parse import format_item_for_prompt
        from efm.hparams import EFMHParams

        tid = args.task_name.split(".")[-1]
        cur_query = json.load(open(f"./config_files/{tid}.json"))["intent"]

        items = store.load_json(f"{args.efm_dir}/items.json", {})
        experiences = store.load_json(f"{args.efm_dir}/experiences.json", {})
        now = store.load_json(f"{args.efm_dir}/now.json", {"now": 0})["now"]
        hp = EFMHParams(k_cand=args.efm_k_cand, eta=args.efm_eta, gamma=args.efm_gamma)

        injected = retrieve(
            query=cur_query, now=now, task_id=tid, items=items, experiences=experiences, hp=hp,
            query_cache_path=f"{args.efm_dir}/query_embeddings.jsonl",
            desc_cache_path=f"{args.efm_dir}/desc_embeddings.jsonl",
            n_inject=args.efm_n_inject,
        )
        store.save_json(f"{args.efm_dir}/items.json", items)  # persist n_retrieved bumps

        mem_items = [format_item_for_prompt(n, item) for n, (_iid, item) in enumerate(injected, start=1)]
        with open(args.memory_path, "w") as f:
            f.write("\n\n".join(mem_items) + ("\n" if mem_items else ""))

        # display-order item_ids, so the post-step in pipeline_memory.py can map the agent's
        # "Memory Item N" usage declaration back to item_ids without re-deriving retrieval.
        store.save_json(f"{args.efm_dir}/last_injected.json", [iid for iid, _item in injected])

        if args.usage_log_path:
            # Which source experience(s) (= originating task_id) each injected item came from --
            # directly comparable to baseline's log_retrieval_event() output, used by the
            # equivalence test (EFM_TASK.md 5.1) to check both modes picked the same source.
            source_task_ids = []
            for iid, _item in injected:
                for ex_id, exp in experiences.items():
                    if iid in exp["item_ids"]:
                        source_task_ids.append(exp["trajectory_ref"])
                        break
            log_retrieval_event(args.usage_log_path, query_task_id=tid, retrieved_task_ids=source_task_ids)

    elif args.reme_dir:
        ensure_file(args.memory_path)
        os.makedirs(args.reme_dir, exist_ok=True)

        from reme import experience as reme_store
        from reme.retrieve import retrieve as reme_retrieve
        from reme.extract import format_item_for_prompt as reme_format_item

        tid = args.task_name.split(".")[-1]
        cur_query = json.load(open(f"./config_files/{tid}.json"))["intent"]

        items = reme_store.load_items(f"{args.reme_dir}/items.json")
        now = reme_store.load_items(f"{args.reme_dir}/now.json").get("now", 0) if os.path.exists(
            f"{args.reme_dir}/now.json") else 0

        injected = reme_retrieve(
            query=cur_query, task_id=tid, items=items, now=now, top_k=args.reme_top_k,
            scenario_cache_path=f"{args.reme_dir}/scenario_embeddings.jsonl",
            query_cache_path=f"{args.reme_dir}/query_embeddings.jsonl",
        )
        reme_store.save_items(f"{args.reme_dir}/items.json", items)  # persist n_retrieved bumps

        mem_items = [reme_format_item(n, item) for n, (_iid, item) in enumerate(injected, start=1)]
        with open(args.memory_path, "w") as f:
            f.write("\n\n".join(mem_items) + ("\n" if mem_items else ""))

        # display-order ids, so the post-step in pipeline_memory.py can credit utility
        # (ReMe's u/f) without re-deriving retrieval.
        reme_store.save_items(f"{args.reme_dir}/last_injected.json",
                              {"ids": [iid for iid, _item in injected]})

    elif args.memp_dir:
        ensure_file(args.memory_path)
        os.makedirs(args.memp_dir, exist_ok=True)

        from memp import procedure as memp_store
        from memp.retrieve import retrieve as memp_retrieve

        tid = args.task_name.split(".")[-1]
        cur_query = json.load(open(f"./config_files/{tid}.json"))["intent"]

        procs = memp_store.load_procedures(f"{args.memp_dir}/procedures.json")
        now = memp_store.load_procedures(f"{args.memp_dir}/now.json").get("now", 0) if os.path.exists(
            f"{args.memp_dir}/now.json") else 0

        injected = memp_retrieve(
            query=cur_query, task_id=tid, procs=procs, now=now, top_k=args.memp_top_k,
            key_cache_path=f"{args.memp_dir}/key_embeddings.jsonl",
            query_cache_path=f"{args.memp_dir}/query_embeddings.jsonl",
        )
        memp_store.save_procedures(f"{args.memp_dir}/procedures.json", procs)

        mem_items = [
            memp_store.format_for_prompt(n, proc, args.memp_build)
            for n, (_pid, proc) in enumerate(injected, start=1)
        ]
        with open(args.memory_path, "w") as f:
            f.write("\n\n".join(mem_items) + ("\n" if mem_items else ""))

        memp_store.save_procedures(f"{args.memp_dir}/last_injected.json",
                                   {"ids": [pid for pid, _proc in injected]})

    elif args.memory_path:
        ensure_file(args.memory_path)

        # Agent with memory retrieval

        website = args.memory_path.split("/")[-1].split(".")[0]
        mem_partial_path = args.memory_path.split("/")[0]

        if not os.path.exists(f"./{mem_partial_path}/{website}.jsonl"):
            open(f"./{mem_partial_path}/{website}.jsonl", "w").close()

        with open(f"./{mem_partial_path}/{website}.jsonl", "r") as f:
            reasoning_bank = [json.loads(line) for line in f.readlines()]

        cur_query = json.load(open(f"./config_files/{args.task_name.split('.')[-1]}.json"))["intent"]

        res = select_memory(n=1,
                            reasoning_bank=reasoning_bank,
                            cur_query=cur_query,
                            task_id=args.task_name.split('.')[-1],
                            cache_path=f"./{mem_partial_path}/{website}_embeddings.jsonl",
                            prefer_model="gemini")

        if args.usage_log_path:
            log_retrieval_event(
                args.usage_log_path,
                query_task_id=args.task_name.split('.')[-1],
                retrieved_task_ids=[item["task_id"] for item in res],
            )

        if not res:
            with open(args.memory_path, "w") as f:
                f.write("")
        else:
            mem_items = []
            for item in res:
                for i in item["memory_items"]:
                    mem_items.append(i)
            with open(args.memory_path, "w") as f:
                f.write("\n\n".join(mem_items) + "\n")

    env_args = EnvArgs(
        task_name=args.task_name,
        task_seed=None,
        max_steps=args.max_steps,
        headless=args.headless,
        record_video=args.record_video,
        viewport={"width": 1500, "height": 1280},
        slow_mo=args.slow_mo,
    )

    if args.task_name == "openended":
        env_args.wait_for_user_message = True
        env_args.task_kwargs = {"start_url": args.start_url}

    exp_args = ExpArgs(
        env_args=env_args,
        logging_level=logging.DEBUG if args.verbose else logging.INFO,
        logging_level_stdout=logging.DEBUG if args.verbose else logging.INFO,
        agent_args=GenericAgentArgs(
            chat_model_args=ChatModelArgs(
                model_name=args.model_name,
                temperature=0.7,
                max_total_tokens=128_000,  # "Maximum total tokens for the chat model."
                max_input_tokens=126_000,  # "Maximum tokens for the input to the chat model."
                max_new_tokens=65_536,  # "Maximum total tokens for the chat model."
            ),
            flags=Flags(
                use_html=args.use_html,
                use_ax_tree=args.use_ax_tree,
                use_thinking=args.use_thinking,  # "Enable the agent with a memory (scratchpad)."
                use_error_logs=True,  # "Prompt the agent with the error logs."
                use_memory=False,  # "Enables the agent with a memory (scratchpad)."
                use_history=args.use_history,
                use_diff=False,  # "Prompt the agent with the difference between the current and past observation."
                use_past_error_logs=True,  # "Prompt the agent with the past error logs."
                use_action_history=True,  # "Prompt the agent with the action history."
                multi_actions=args.multi_actions,
                use_abstract_example=True,  # "Prompt the agent with an abstract example."
                use_concrete_example=True,  # "Prompt the agent with a concrete example."
                use_screenshot=args.use_screenshot,
                enable_chat=True,
                demo_mode="default" if args.demo_mode else "off",
                memory_path=args.memory_path,
            ),
        ),
    )

    exp_args.prepare(Path(args.results_path))
    exp_args.run()

    dest = os.path.join(args.results_path, args.task_name)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    for _ in range(10):
        try:
            os.rename(exp_args.exp_dir, dest)
            break
        except PermissionError:
            time.sleep(1)
    else:
        shutil.copytree(exp_args.exp_dir, dest, dirs_exist_ok=True)

if __name__ == "__main__":
    main()
