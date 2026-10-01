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

"""MaTTS parallel scaling pipeline for WebArena (paper Section 3.3): for each task,
run k independent trajectories concurrently -- each against its OWN live site
container, since WebArena carries mutable state (cart, order history, ...) that would
contaminate a shared instance -- then use an LLM Best-of-N judge (scaling.py, the
paper's own prompt) to pick the single best one. Only that selected trajectory is
scored (autoeval) and used for memory induction.

Requires k site containers per website already running and base-url-configured (see
Setup.md's container clone recipe): shopping on 7770/7771/7772, shopping_admin on
7780/7781/7782.

Usage (from WebArena/):
    python pipeline_matts.py --website shopping --model gemini-3.5-flash-lite \
        --output_dir results_shopping_matts_gemini-3.5-flash-lite --k 3 --end_index 15
"""

import os
import json
import argparse
import shutil
import signal
import subprocess
import sys
import time
from subprocess import Popen

from utils.clients import CLIENT_DICT
import scaling as scaling_lib

SITE_PORTS = {
    "shopping": [7770, 7771, 7772],
    "shopping_admin": [7780, 7781, 7782],
}


def env_for_trial(website: str, i: int) -> dict:
    env = os.environ.copy()
    port = SITE_PORTS[website][i]
    if website == "shopping":
        env["WA_SHOPPING"] = f"http://localhost:{port}"
    elif website == "shopping_admin":
        env["WA_SHOPPING_ADMIN"] = f"http://localhost:{port}/admin"
    return env


def newest_mtime(paths: list[str]) -> float:
    newest = 0.0
    for path in paths:
        if not os.path.exists(path):
            continue
        for root, _, files in os.walk(path):
            for name in files:
                try:
                    newest = max(newest, os.path.getmtime(os.path.join(root, name)))
                except OSError:
                    pass
    return newest


def stop_process_tree(process: Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
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


def wait_group_with_idle_timeout(procs: list[Popen], watch_dirs: list[str], idle_timeout: int, label: str) -> None:
    """Waits for all procs to finish; kills the whole group if no result-file activity
    across any watch_dir for idle_timeout seconds."""
    last_activity = time.monotonic()
    last_mtime = newest_mtime(watch_dirs)
    while any(p.poll() is None for p in procs):
        time.sleep(5)
        current_mtime = newest_mtime(watch_dirs)
        if current_mtime > last_mtime:
            last_mtime = current_mtime
            last_activity = time.monotonic()
        if time.monotonic() - last_activity >= idle_timeout:
            print(f"[matts] {label}: no activity for {idle_timeout}s; killing all trials.", flush=True)
            for p in procs:
                stop_process_tree(p)
            return


def run_with_idle_timeout(cmd: list[str], watch_dir: str, idle_timeout: int, label: str) -> int | None:
    process = Popen(cmd)
    last_activity = time.monotonic()
    last_mtime = newest_mtime([watch_dir])
    while process.poll() is None:
        time.sleep(5)
        current_mtime = newest_mtime([watch_dir])
        if current_mtime > last_mtime:
            last_mtime = current_mtime
            last_activity = time.monotonic()
        if time.monotonic() - last_activity >= idle_timeout:
            print(f"[matts] {label}: no activity for {idle_timeout}s; killing.", flush=True)
            stop_process_tree(process)
            return None
    return process.returncode


def record_failure(output_dir: str, task_id: str, stage: str, reason: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "pipeline_failures.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "task_id": task_id, "stage": stage, "reason": reason,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, ensure_ascii=False) + "\n")


def main():
    if args.k > len(SITE_PORTS[args.website]):
        raise ValueError(
            f"k={args.k} exceeds configured site containers for {args.website} "
            f"({len(SITE_PORTS[args.website])} available). Add more ports to SITE_PORTS."
        )
    memory_dir = args.memory_dir or f"memories_{args.memory_mode}"
    trials_root = os.path.join(args.output_dir, "_trials")

    config_files = [
        os.path.join("config_files", f) for f in os.listdir("config_files")
        if f.endswith(".json") and f.split(".")[0].isdigit()
    ]
    config_files = sorted(config_files, key=lambda x: int(os.path.basename(x).split(".")[0]))
    config_list = [json.load(open(f)) for f in config_files]
    config_flags = [config["sites"] == [args.website] for config in config_list]
    task_configs = [config for config, flag in zip(config_list, config_flags) if flag]

    if args.end_index is None:
        args.end_index = len(task_configs)

    bon_client = CLIENT_DICT[args.model](model_name=args.model)

    for config in task_configs[args.start_index:args.end_index]:
        tid = config["task_id"]
        if int(tid) <= args.prev_id:
            continue

        # forward slash on purpose: autoeval/evaluate_trajectory.py parses the task id
        # via result_dir.split('/')[-1].split('.')[1], which breaks on Windows backslash
        # paths (and output_dir names here contain a literal '.' from e.g. "3.5")
        canonical_dir = f"{args.output_dir}/webarena.{tid}"
        if os.path.isdir(canonical_dir) and not args.rerun_completed:
            print(f"[matts] task {tid}: already complete; skipping", flush=True)
            continue

        # step 1: k parallel trials, each against its own site container
        print(f"[matts] task {tid}: launching {args.k} parallel trials", flush=True)
        procs = []
        trial_dirs = []
        for i in range(args.k):
            trial_output = os.path.join(trials_root, f"trial_{i}")
            run_cmd = [
                sys.executable, "run.py",
                "--task_name", f"webarena.{tid}",
                "--model_name", args.model,
                "--results_path", trial_output,
                "--headless", "False",
            ]
            if args.memory_mode != "no_memory":
                run_cmd += ["--memory_path", f"{memory_dir}/{args.website}.txt"]
            p = Popen(run_cmd, env=env_for_trial(args.website, i))
            procs.append(p)
            trial_dirs.append(os.path.join(trial_output, f"webarena.{tid}"))

        wait_group_with_idle_timeout(procs, trial_dirs, args.idle_timeout, f"task {tid} trials")
        for p in procs:
            stop_process_tree(p)  # no-op if already exited

        valid_dirs = [d for d in trial_dirs if os.path.isdir(d)]
        if not valid_dirs:
            record_failure(args.output_dir, tid, "inference", "all trials failed")
            print(f"[matts] task {tid}: all {args.k} trials failed; skipping.", flush=True)
            continue

        # step 2: Best-of-N selection among whichever trials actually completed
        best_idx, bon_raw = scaling_lib.select_best_trajectory(bon_client, config["intent"], valid_dirs)
        best_dir = valid_dirs[best_idx]
        print(f"[matts] task {tid}: selected trial {best_idx} of {len(valid_dirs)}", flush=True)

        if os.path.isdir(canonical_dir):
            shutil.rmtree(canonical_dir)
        shutil.copytree(best_dir, canonical_dir)
        with open(os.path.join(canonical_dir, "scaling_info.json"), "w", encoding="utf-8") as f:
            json.dump({
                "k": args.k, "num_valid_trials": len(valid_dirs),
                "selected_index": best_idx, "bon_raw": bon_raw,
            }, f, ensure_ascii=False, indent=2)
        for d in trial_dirs:
            if os.path.isdir(d):
                shutil.rmtree(d)

        # step 3: autoeval on the selected trajectory
        eval_cmd = [
            sys.executable, "-m", "autoeval.evaluate_trajectory",
            "--result_dir", canonical_dir,
            "--model", args.model,
            "--log_dir", f"autoeval/logs_matts_{args.website}",
        ]
        returncode = run_with_idle_timeout(eval_cmd, canonical_dir, args.idle_timeout, f"task {tid} autoeval")
        if returncode != 0:
            record_failure(args.output_dir, tid, "autoeval", "idle timeout" if returncode is None else f"exit code {returncode}")
            print(f"[matts] task {tid}: autoeval failed; skipping memory induction.", flush=True)
            continue

        if args.memory_mode == "no_memory":
            continue

        # step 4: memory induction from the selected trajectory only
        memory_cmd = [
            sys.executable, "induce_memory.py",
            "--result_dir", args.output_dir,
            "--task", f"webarena.{tid}",
            "--criteria", args.judge,
            "--memory_mode", args.memory_mode,
            "--model", args.model,
            "--output_path", f"{memory_dir}/{args.website}.jsonl",
        ]
        returncode = run_with_idle_timeout(memory_cmd, canonical_dir, args.idle_timeout, f"task {tid} memory induction")
        if returncode != 0:
            record_failure(args.output_dir, tid, "memory", "idle timeout" if returncode is None else f"exit code {returncode}")
            print(f"[matts] task {tid}: memory induction failed.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--website", type=str, required=True, choices=["shopping", "shopping_admin"])
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="gemini-3.5-flash-lite",
                         choices=["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "claude-3-7-sonnet@20250219", "gemini-2.5-pro", "google/gemma-3-12b-it"])
    parser.add_argument("--k", type=int, default=3, help="Number of parallel trajectories per task.")
    parser.add_argument("--prev_id", type=int, default=-1)
    parser.add_argument("--rerun_completed", action="store_true")
    parser.add_argument("--memory_mode", type=str, default="reasoningbank", choices=["no_memory", "reasoningbank"])
    parser.add_argument("--memory_dir", type=str, default=None)
    parser.add_argument("--judge", type=str, default="autoeval")
    parser.add_argument("--idle_timeout", type=int, default=300)
    args = parser.parse_args()

    main()
