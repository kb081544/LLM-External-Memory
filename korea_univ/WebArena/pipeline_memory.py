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
    is_ace = args.memory_mode == "ace"
    is_reme = args.memory_mode == "reme"
    is_memp = args.memory_mode == "memp"
    is_cer = args.memory_mode == "cer"
    prune_counter = 0
    efm_hp = EFMHParams.from_args(args) if is_efm else None
    efm_dir = f"{memory_dir}/{args.website}" if is_efm else None
    efm_client = CLIENT_DICT[args.model](model_name=args.model) if is_efm else None
    ace_dir = f"{memory_dir}/{args.website}" if is_ace else None
    ace_playbook_path = f"{ace_dir}/playbook.json" if is_ace else None
    ace_cache_path = f"{ace_dir}/content_embeddings.jsonl" if is_ace else None
    reme_dir = f"{memory_dir}/{args.website}" if is_reme else None
    memp_dir = f"{memory_dir}/{args.website}" if is_memp else None
    cer_dir = f"{memory_dir}/{args.website}" if is_cer else None
    arm_client = (
        CLIENT_DICT[args.model](model_name=args.model) if (is_reme or is_memp) else None
    )

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
        if is_ace:
            run_cmd += ["--ace_dir", ace_dir, "--ace_max_bullets", str(args.ace_max_bullets)]
        if is_reme:
            run_cmd += ["--reme_dir", reme_dir, "--reme_top_k", str(args.reme_top_k)]
        if is_memp:
            run_cmd += [
                "--memp_dir", memp_dir,
                "--memp_top_k", str(args.memp_top_k),
                "--memp_build", args.memp_build,
            ]
        if is_cer:
            run_cmd += [
                "--cer_dir", cer_dir,
                "--cer_max_dynamics", str(args.cer_max_dynamics),
                "--cer_max_skills", str(args.cer_max_skills),
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

        if is_reme:
            # ReMe's utility bookkeeping (paper 3.4): u increments only when a retrieved
            # experience was in play on a task the judge called a success. Done here, before
            # extraction, so the credit is attributed to the bank state the agent actually saw.
            from reme import experience as reme_store
            from reme.utility import credit_outcome as reme_credit, prune as reme_prune

            now = reme_store.load_items(f"{reme_dir}/now.json").get("now", 0)
            items = reme_store.load_items(f"{reme_dir}/items.json")
            injected_ids = reme_store.load_items(f"{reme_dir}/last_injected.json").get("ids", [])
            injected = [(iid, items[iid]) for iid in injected_ids if iid in items]
            judged_success = read_task_outcome(args.output_dir, tid, args.judge, args.model)
            reme_credit(injected, judged_success=judged_success, now=now,
                        events_path=f"{reme_dir}/reme_events.jsonl", task_id=tid)
            num_deleted = reme_prune(items, now=now, events_path=f"{reme_dir}/reme_events.jsonl",
                                     alpha=args.reme_alpha, beta=args.reme_beta)
            reme_store.save_items(f"{reme_dir}/items.json", items)
            if num_deleted:
                print(f"[pipeline] reme deleted {num_deleted} low-utility experience(s) "
                      f"after task {tid}.", flush=True)

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
            "--output_path", (
                f"{efm_dir}/_raw.jsonl" if is_efm
                else f"{ace_dir}/_raw.jsonl" if is_ace
                else f"{reme_dir}/_raw.jsonl" if is_reme
                else f"{memp_dir}/_raw.jsonl" if is_memp
                else f"{cer_dir}/_raw.jsonl" if is_cer
                else f"{memory_dir}/{args.website}.jsonl"
            ),
        ]
        if is_cer:
            # The distillation prompts are shown the buffer so they can answer
            # "Summarized before" rather than re-distilling (CER's own dedup, paper 3.1).
            memory_cmd += ["--cer_dir", cer_dir, "--cer_page_chars", str(args.cer_page_chars)]
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

        elif is_ace:
            from ace.playbook import load_playbook, save_playbook
            from ace.curate import parse_candidate_bullets, curate

            with open(f"{ace_dir}/_raw.jsonl", encoding="utf-8") as f:
                raw_entry = json.loads(f.readlines()[-1])
            candidates = parse_candidate_bullets("\n\n".join(raw_entry["memory_items"]))
            playbook = load_playbook(ace_playbook_path)
            num_added = curate(
                candidates, playbook, task_succeeded=(raw_entry["status"] == "success"),
                cache_path=ace_cache_path,
            )
            save_playbook(ace_playbook_path, playbook)
            print(
                f"[pipeline] ace: {len(candidates)} candidate bullet(s), {num_added} new "
                f"(rest merged as helpful/harmful votes) after task {tid}.", flush=True,
            )

        elif is_reme:
            from reme import experience as reme_store
            from reme.extract import parse_experiences
            from reme.compare import find_counterpart, comparative_extract
            from efm import store as efm_store_mod

            with open(f"{reme_dir}/_raw.jsonl", encoding="utf-8") as f:
                raw_entry = json.loads(f.readlines()[-1])
            now = reme_store.load_items(f"{reme_dir}/now.json").get("now", 0)
            items = reme_store.load_items(f"{reme_dir}/items.json")

            parsed = parse_experiences("\n\n".join(raw_entry["memory_items"]))
            facets = [("success" if raw_entry["status"] == "success" else "failure", p)
                      for p in parsed]

            # Third analysis: comparative, only when this task has a success/fail counterpart
            # of the same intent_template_id (see reme/compare.py).
            counterpart = find_counterpart(
                f"{reme_dir}/_raw.jsonl", raw_entry.get("template_id"), raw_entry["status"], tid,
            )
            if counterpart is not None:
                try:
                    comp_raw = comparative_extract(arm_client, raw_entry, counterpart)
                    facets += [("comparative", p) for p in parse_experiences(comp_raw)]
                except Exception as exc:
                    print(f"[pipeline] reme comparative extraction failed: "
                          f"{type(exc).__name__}: {exc}", flush=True)

            new_ids = []
            for facet, parsed_item in facets:
                iid = reme_store.next_id(items)
                items[iid] = reme_store.new_item(parsed_item, task_id=tid,
                                                 status=raw_entry["status"], now=now, facet=facet)
                # Retrieval is indexed on the usage scenario (paper 4.3), so that is what gets
                # embedded into the scenario cache.
                efm_store_mod.embed_and_cache(
                    parsed_item["scenario"], f"{reme_dir}/scenario_embeddings.jsonl", iid,
                )
                new_ids.append(iid)

            reme_store.save_items(f"{reme_dir}/items.json", items)
            reme_store.save_items(f"{reme_dir}/now.json", {"now": now + 1})
            n_comp = sum(1 for facet, _ in facets if facet == "comparative")
            print(f"[pipeline] reme stored {len(new_ids)} experience(s) "
                  f"({n_comp} comparative) after task {tid}.", flush=True)

        elif is_memp:
            from memp import procedure as memp_store
            from memp.update import apply_update
            from efm import store as efm_store_mod

            with open(f"{memp_dir}/_raw.jsonl", encoding="utf-8") as f:
                raw_entry = json.loads(f.readlines()[-1])
            now = memp_store.load_procedures(f"{memp_dir}/now.json").get("now", 0)
            procs = memp_store.load_procedures(f"{memp_dir}/procedures.json")
            injected_ids = memp_store.load_procedures(
                f"{memp_dir}/last_injected.json").get("ids", [])
            injected = [(pid, procs[pid]) for pid in injected_ids if pid in procs]

            parsed = memp_store.parse_procedure("\n\n".join(raw_entry["memory_items"]))
            trajectory_text = "\n\n".join(
                f"<think>\n{t}\n</think>\n<action>\n{a}\n</action>"
                for t, a in zip(raw_entry.get("think_list", []), raw_entry.get("action_list", []))
            )
            judged_success = raw_entry["status"] == "success"

            new_pid, adjusted = apply_update(
                args.memp_update, procs, parsed, judged_success=judged_success,
                injected=injected, query=raw_entry["query"], trajectory=trajectory_text,
                task_id=tid, now=now, client=arm_client,
                events_path=f"{memp_dir}/memp_events.jsonl",
            )
            if new_pid is not None:
                # Key=Query retrieval (paper 4.2): the source task's query is the index key.
                efm_store_mod.embed_and_cache(
                    raw_entry["query"], f"{memp_dir}/key_embeddings.jsonl", new_pid,
                )
            if adjusted:
                # The revised procedure keeps its key embedding: Memp's Adjustment rewrites
                # the procedure body, not what task class it answers.
                pass

            memp_store.save_procedures(f"{memp_dir}/procedures.json", procs)
            memp_store.save_procedures(f"{memp_dir}/now.json", {"now": now + 1})
            action = ("appended " + new_pid) if new_pid else ("adjusted in place" if adjusted
                                                              else "no change")
            print(f"[pipeline] memp ({args.memp_update}): {action} after task {tid}.", flush=True)

        elif is_cer:
            from cer.buffer import load_buffer, save_buffer
            from cer.distill import parse_dynamics, parse_skills, merge_dynamics, merge_skills

            with open(f"{cer_dir}/_raw.jsonl", encoding="utf-8") as f:
                raw_entry = json.loads(f.readlines()[-1])

            buffer_path = f"{cer_dir}/buffer.json"
            buf = load_buffer(buffer_path)
            now = len(buf["dynamics"]) + len(buf["skills"])  # monotone counter, only used for provenance
            added_d = merge_dynamics(
                parse_dynamics(raw_entry.get("cer_dynamics_raw", "")), buf, tid, now)
            added_s = merge_skills(
                parse_skills(raw_entry.get("cer_skills_raw", "")), buf, tid, now)
            save_buffer(buffer_path, buf)
            print(
                f"[pipeline] cer: +{len(added_d)} dynamics, +{len(added_s)} skills "
                f"(buffer now {len(buf['dynamics'])}/{len(buf['skills'])}) after task {tid}.",
                flush=True,
            )


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
                        choices=["no_memory", "reasoningbank", "awm", "synapse", "reasoningbank_pruned", "efm", "ace", "reme", "memp", "cer"])
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
    parser.add_argument(
        "--ace_max_bullets",
        type=int,
        default=20,
        help="ace only: cap on how many playbook bullets get injected per task (ranked by "
             "helpful-harmful utility).",
    )
    parser.add_argument(
        "--reme_top_k",
        type=int,
        default=3,
        help="reme only: how many experiences to inject per task (ReMe retrieves top-k by "
             "usage-scenario similarity).",
    )
    parser.add_argument(
        "--reme_alpha",
        type=int,
        default=5,
        help="reme only: minimum retrievals before an experience can be deleted (paper's alpha).",
    )
    parser.add_argument(
        "--reme_beta",
        type=float,
        default=0.5,
        help="reme only: delete when u/f <= beta once f >= alpha (paper's beta).",
    )
    parser.add_argument(
        "--memp_top_k",
        type=int,
        default=1,
        help="memp only: how many procedures to inject per task.",
    )
    parser.add_argument(
        "--memp_build",
        type=str,
        default="proceduralization",
        choices=["script", "trajectory", "proceduralization"],
        help="memp only: which BUILD condition to inject (paper 4.2).",
    )
    parser.add_argument(
        "--memp_update",
        type=str,
        default="adjustment",
        choices=["vanilla", "validation", "adjustment"],
        help="memp only: which UPDATE strategy to apply (paper 4.3). 'adjustment' is the one "
             "that rewrites an existing procedure in place on failure.",
    )
    parser.add_argument(
        "--cer_max_dynamics",
        type=int,
        default=5,
        help="cer only: k_d, max page/URL experiences replayed per task (paper 4.1.1 uses 5).",
    )
    parser.add_argument(
        "--cer_max_skills",
        type=int,
        default=5,
        help="cer only: k_s, max skill experiences replayed per task (paper 4.1.1 uses 5).",
    )
    parser.add_argument(
        "--cer_page_chars",
        type=int,
        default=1500,
        help="cer only: per-step accessibility-tree slice the dynamics distillation module "
             "sees. CER is the one arm that summarizes pages, so it needs observations.",
    )
    EFMHParams.add_cli_args(parser)
    args = parser.parse_args()

    main()
