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
import json
import argparse
import signal
import subprocess
import time
from subprocess import Popen
import random
import sys

from memory_prune import update_usage_stats, prune_memory_bank
from efm.hparams import EFMHParams
from efm import store as efm_store
from efm.usage import record_usage as efm_record_usage
from efm.insert import insert as efm_insert_fn
from efm.forget import forget as efm_forget_fn
from efm.parse import parse_memory_items
from induce_memory import extract_think_and_action
from utils.clients import CLIENT_DICT


def newest_result_mtime(path):
    """Return the newest mtime below path without failing on transient files."""
    newest = 0.0
    if not path or not os.path.exists(path):
        return newest
    for root, _, files in os.walk(path):
        for name in files:
            try:
                newest = max(newest, os.path.getmtime(os.path.join(root, name)))
            except OSError:
                pass
    return newest


def stop_process_tree(process):
    """Stop run.py and its Playwright/Chromium children after a hang."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_with_idle_timeout(cmd, results_dir, idle_timeout, label):
    """Run a command, aborting its process tree if result files stop changing."""
    popen_kwargs = {"start_new_session": True} if os.name != "nt" else {}
    process = Popen(cmd, **popen_kwargs)
    last_activity = time.monotonic()
    last_mtime = newest_result_mtime(results_dir)

    while process.poll() is None:
        time.sleep(5)
        current_mtime = newest_result_mtime(results_dir)
        if current_mtime > last_mtime:
            last_mtime = current_mtime
            last_activity = time.monotonic()
        if time.monotonic() - last_activity >= idle_timeout:
            print(
                f"[watchdog] {label} produced no result activity for "
                f"{idle_timeout}s; terminating PID {process.pid}.",
                flush=True,
            )
            stop_process_tree(process)
            return None
    return process.returncode


def record_failure(output_dir, task_id, stage, reason):
    """Keep skipped tasks visible without adding them to the memory bank."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "pipeline_failures.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "task_id": task_id,
            "stage": stage,
            "reason": reason,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, ensure_ascii=False) + "\n")


def read_task_outcome(output_dir, task_id, judge, model):
    """Same success/fail signal induce_memory.py itself uses (gt cum_reward or autoeval
    rm), read independently here so usage-stat crediting doesn't depend on this task's own
    memory induction having run yet."""
    cur_task = os.path.join(output_dir, f"webarena.{task_id}")
    if judge == "gt":
        reward = json.load(open(os.path.join(cur_task, "summary_info.json")))["cum_reward"]
    else:
        reward = json.load(open(os.path.join(cur_task, f"{model}_autoeval.json")))[0]["rm"]
    return reward == 1


def main():
    memory_dir = args.memory_dir or f"memories_{args.memory_mode}"
    is_pruned = args.memory_mode == "reasoningbank_pruned"
    is_efm = args.memory_mode == "efm"
    prune_counter = 0
    efm_hp = EFMHParams.from_args(args) if is_efm else None
    efm_dir = f"{memory_dir}/{args.website}" if is_efm else None
    efm_client = CLIENT_DICT[args.model](model_name=args.model) if is_efm else None

    # collect examples
    config_files = [
        os.path.join("config_files", f) for f in os.listdir("config_files")
        if f.endswith(".json") and f.split(".")[0].isdigit()
    ]
    config_files = sorted(config_files, key=lambda x: int(os.path.basename(x).split(".")[0]))
    config_list = [json.load(open(f)) for f in config_files]
    if args.website == "multi":
        config_flags = []
        for config in config_list:
            if len(config["sites"]) != 1 and "map" not in config['sites']:
                config_flags.append(True)
            else:
                config_flags.append(False)
    else:
        config_flags = [config["sites"] == [args.website] for config in config_list]
    task_ids = [config["task_id"] for config, flag in zip(config_list, config_flags) if flag]
    # random.shuffle(task_ids)

    if args.end_index == None: 
        args.end_index = len(task_ids)

    # num = args.output_dir.split("_")[-1]

    for tid in task_ids[args.start_index: args.end_index]:

        if int(tid) <= args.prev_id:
            continue

        # step 1: run inference
        run_cmd = [
            sys.executable, "run.py",
            "--task_name", f"webarena.{tid}",
            "--model_name", args.model,
            "--results_path", f"{args.output_dir}",
            "--headless", "True",
            # demo_mode/slow_mo exist purely for human-watched visual demos (cursor highlight
            # effects, artificial per-action delay) -- pure wasted wall-clock time in an unheaded,
            # unattended batch run with nobody watching. No effect on action correctness/results.
            "--demo_mode", "False",
            "--slow_mo", "0",
        ]
        if args.memory_mode != "no_memory":
            run_cmd += ["--memory_path", f"{memory_dir}/{args.website}.txt"]
        if is_pruned:
            run_cmd += ["--usage_log_path", f"{memory_dir}/{args.website}_usage.jsonl"]
        if is_efm:
            run_cmd += [
                "--efm_dir", efm_dir,
                "--efm_k_cand", str(efm_hp.k_cand),
                "--efm_eta", str(efm_hp.eta),
                "--efm_gamma", str(efm_hp.gamma),
                "--efm_n_inject", str(args.efm_n_inject),
                "--usage_log_path", f"{memory_dir}/{args.website}_usage.jsonl",
            ]
        run_ok = False
        for attempt in range(1, args.run_retries + 2):
            print(
                f"[pipeline] task {tid}: inference attempt "
                f"{attempt}/{args.run_retries + 1}",
                flush=True,
            )
            returncode = run_with_idle_timeout(
                run_cmd,
                args.output_dir,
                args.idle_timeout,
                f"task {tid} inference",
            )
            if returncode == 0 and os.path.isdir(
                os.path.join(args.output_dir, f"webarena.{tid}")
            ):
                run_ok = True
                break
            reason = "idle timeout" if returncode is None else f"exit code {returncode}"
            print(f"[pipeline] task {tid}: {reason}.", flush=True)

        if not run_ok:
            record_failure(args.output_dir, tid, "inference", reason)
            print(
                f"[pipeline] task {tid}: retries exhausted; skipping evaluation "
                "and memory induction.",
                flush=True,
            )
            continue

        # step 2: run evaluation
        eval_cmd = [
            sys.executable, "-m", "autoeval.evaluate_trajectory",
            "--result_dir", f"{args.output_dir}/webarena.{tid}",
            "--model", args.model,
            "--log_dir", f"autoeval/logs_{args.memory_mode}_{args.website}",
        ]
        returncode = run_with_idle_timeout(
            eval_cmd,
            args.output_dir,
            args.idle_timeout,
            f"task {tid} autoeval",
        )
        if returncode != 0:
            reason = "idle timeout" if returncode is None else f"exit code {returncode}"
            record_failure(args.output_dir, tid, "autoeval", reason)
            print(
                f"[pipeline] task {tid}: autoeval failed; skipping memory induction.",
                flush=True,
            )
            continue

        if is_pruned:
            stats_path = f"{memory_dir}/{args.website}_stats.json"
            success = read_task_outcome(args.output_dir, tid, args.judge, args.model)
            update_usage_stats(
                stats_path=stats_path,
                usage_log_path=f"{memory_dir}/{args.website}_usage.jsonl",
                outcome_task_id=str(tid),
                success=success,
            )
            prune_counter += 1
            if prune_counter % args.prune_every == 0 and os.path.exists(stats_path):
                with open(stats_path) as f:
                    stats = json.load(f)
                num_pruned = prune_memory_bank(f"{memory_dir}/{args.website}.jsonl", stats)
                if num_pruned:
                    print(
                        f"[pipeline] pruned {num_pruned} low-utility memory entries "
                        f"after task {tid}.",
                        flush=True,
                    )

        if is_efm:
            now = efm_store.load_json(f"{efm_dir}/now.json", {"now": 0})["now"]
            injected_ids = efm_store.load_json(f"{efm_dir}/last_injected.json", [])
            items = efm_store.load_json(f"{efm_dir}/items.json", {})
            injected = [(iid, items[iid]) for iid in injected_ids if iid in items]
            think_list, _actions = extract_think_and_action(os.path.join(args.output_dir, f"webarena.{tid}"))
            judged_success = read_task_outcome(args.output_dir, tid, args.judge, args.model)
            efm_record_usage(
                injected=injected, think_text="\n\n".join(think_list), now=now, client=efm_client,
                events_path=f"{efm_dir}/efm_events.jsonl", task_id=tid, judged_success=judged_success,
            )
            efm_store.save_json(f"{efm_dir}/items.json", items)

        if args.memory_mode == "no_memory":
            continue

        # step 3: extract new memory items
        memory_cmd = [
            sys.executable, "induce_memory.py",
            "--result_dir", args.output_dir,
            "--task", f"webarena.{tid}",
            "--criteria", args.judge,
            "--memory_mode", "reasoningbank" if (is_pruned or is_efm) else args.memory_mode,
            "--model", args.model,
            "--output_path", f"{efm_dir}/_raw.jsonl" if is_efm else f"{memory_dir}/{args.website}.jsonl",
        ]
        returncode = run_with_idle_timeout(
            memory_cmd,
            args.output_dir,
            args.idle_timeout,
            f"task {tid} memory induction",
        )
        if returncode != 0:
            reason = "idle timeout" if returncode is None else f"exit code {returncode}"
            record_failure(args.output_dir, tid, "memory", reason)
            print(f"[pipeline] task {tid}: memory induction failed.", flush=True)
        elif is_efm:
            with open(f"{efm_dir}/_raw.jsonl", encoding="utf-8") as f:
                raw_entry = json.loads(f.readlines()[-1])
            new_items_raw = parse_memory_items(raw_entry["memory_items"])
            items = efm_store.load_json(f"{efm_dir}/items.json", {})
            experiences = efm_store.load_json(f"{efm_dir}/experiences.json", {})
            item_ids = efm_insert_fn(
                new_items_raw, task_id=tid, now=now, items=items, hp=efm_hp,
                desc_cache_path=f"{efm_dir}/desc_embeddings.jsonl",
                content_cache_path=f"{efm_dir}/content_embeddings.jsonl",
                client=efm_client, source_label=raw_entry["status"],
            )
            experiences[f"ex_{tid}"] = {
                "query": raw_entry["query"], "trajectory_ref": tid, "item_ids": item_ids,
            }
            efm_store.save_json(f"{efm_dir}/items.json", items)
            efm_store.save_json(f"{efm_dir}/experiences.json", experiences)

            num_forgotten = efm_forget_fn(items, now=now, hp=efm_hp, events_path=f"{efm_dir}/efm_events.jsonl")
            efm_store.save_json(f"{efm_dir}/items.json", items)
            if num_forgotten:
                print(f"[pipeline] efm forgot {num_forgotten} item(s) after task {tid}.", flush=True)

            efm_store.save_json(f"{efm_dir}/now.json", {"now": now + 1})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--website", type=str, required=True,
                        choices=["shopping", "shopping_admin", "gitlab", "reddit", "multi"])
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model", type=str, default="gemini-3.6-flash",
                        choices=["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "claude-3-7-sonnet@20250219", "gemini-2.5-pro", "google/gemma-3-12b-it", "agy-claude-sonnet-4-6", "ccli-sonnet"])
    parser.add_argument("--prev_id", type=int, default=-1)
    parser.add_argument("--memory_mode", type=str, default="reasoningbank",
                        choices=["no_memory", "reasoningbank", "awm", "synapse", "reasoningbank_pruned", "efm"])
    parser.add_argument("--memory_dir", type=str, default=None,
                        help="Override the memory directory (default: memories_<memory_mode>). "
                             "Use to keep a test run's memory bank separate from the main one.")
    parser.add_argument("--judge", type=str, default="autoeval")
    parser.add_argument(
        "--prune_every",
        type=int,
        default=20,
        help="reasoningbank_pruned only: recompute utility and prune low-utility memory "
             "bank entries every N tasks.",
    )
    parser.add_argument(
        "--idle_timeout",
        type=int,
        default=300,
        help="Kill and retry inference after this many seconds without result-file activity.",
    )
    parser.add_argument(
        "--run_retries",
        type=int,
        default=1,
        help="Number of automatic inference retries after a crash or idle timeout.",
    )
    parser.add_argument(
        "--efm_n_inject",
        type=int,
        default=2,
        help="efm only: how many items to inject per task, matched to baseline's *measured* "
             "average injection count (EFM_TASK.md 3.2/5.1) -- set this from the sanity-check "
             "measurement before the real run, not left at the placeholder default.",
    )
    EFMHParams.add_cli_args(parser)
    args = parser.parse_args()

    main()
