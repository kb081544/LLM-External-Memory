"""Run an LLM over a sampled subset of a Mind2Web domain's train tasks (Shopping, Travel,
or Entertainment -- see download_data.py) and build a WebArena-style visual report
(per-task screenshots + HTML report + gallery).

Mind2Web is offline: every task is a fixed sequence of already-captured steps (saved
HTML, a ground-truth target element, and an operation). There's no live site to browse,
so "running the agent" means: at each step, show the model the task, the *ground-truth*
action history so far (standard Mind2Web protocol -- steps are scored independently so
one wrong step doesn't cascade into contaminating later steps' context), and a bounded
candidate shortlist (candidates.py); ask it to pick a candidate + operation + value;
score against the ground truth; render + highlight + screenshot the step.

Usage (run from Mind2Web/):
    python download_data.py    # once, to populate data/
    python run.py --limit 3    # smoke test
    python run.py --limit 30   # full run
    python run.py --report-only  # rebuild reports/summary without calling the LLM
"""

from __future__ import annotations

import argparse
import hashlib
import html as html_lib
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from utils import gemini_retry  # noqa: F401 — patches google.genai with backoff retry
except ImportError:
    pass

from candidates import build_candidate_list, format_candidates_for_prompt, ground_truth_index
from render import Mind2WebRenderer
from multimodal_lookup import MultimodalIndex, parse_bbox
from render_real import highlight_and_save
from utils.clients import CLIENT_DICT
import memory as memory_lib
import scaling as scaling_lib
from efm.hparams import EFMHParams
from efm import store as efm_store
from efm.retrieve import retrieve as efm_retrieve
from efm.insert import insert as efm_insert_fn
from efm.forget import forget as efm_forget_fn
from efm.usage import record_usage as efm_record_usage
from efm.parse import parse_memory_items as efm_parse_memory_items, format_item_for_prompt as efm_format_item

SCHEMA_VERSION = 1

SYSTEM_PROMPT = """You are a web agent solving Mind2Web tasks. You are given a task, the actions already taken, and a numbered list of candidate elements found on the current page. Pick exactly ONE candidate by its number, choose an operation, and (for TYPE/SELECT) a value to enter/select.

Respond with ONLY a JSON object and nothing else:
{"candidate": <number>, "operation": "CLICK" | "TYPE" | "SELECT", "value": "<string, empty for CLICK>", "reason": "<one short sentence>"}
"""


def esc(value: Any) -> str:
    return html_lib.escape("" if value is None else str(value))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def step_seed(annotation_id: str, step_idx: int) -> int:
    digest = hashlib.sha256(f"{annotation_id}:{step_idx}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def build_user_prompt(confirmed_task: str, action_history: list[str], candidate_text: str) -> str:
    history = "\n".join(f"{i + 1}. {a}" for i, a in enumerate(action_history)) or "(none yet)"
    return (
        f"Task: {confirmed_task}\n\n"
        f"Actions taken so far:\n{history}\n\n"
        f"Candidates on the current page:\n{candidate_text}\n"
    )


def parse_model_response(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if "candidate" not in obj or "operation" not in obj:
        return None
    return obj


# ============================================================ data / manifest


def load_tasks(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Run `python download_data.py` first."
        )
    return pd.read_parquet(data_path)


def create_manifest(df: pd.DataFrame, limit: int, seed: int) -> dict:
    ids = sorted(df["annotation_id"].tolist())
    rng = random.Random(seed)
    sampled = rng.sample(ids, min(limit, len(ids)))
    sampled.sort()
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "seed": seed,
        "limit": limit,
        "annotation_ids": sampled,
    }


def create_manifest_from_source(source_path: Path, limit: int) -> dict:
    """Slice the first `limit` annotation_ids off an existing manifest (already sorted),
    so this run's task set is a strict subset of another run's -- e.g. a memory-enabled
    run using exactly the first N tasks of the paired no-memory run's sample, for a
    clean apples-to-apples comparison."""
    source = json.loads(source_path.read_text(encoding="utf-8"))
    sliced_ids = source["annotation_ids"][:limit]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "seed": source["seed"],
        "limit": limit,
        "source_manifest": str(source_path.resolve()),
        "annotation_ids": sliced_ids,
    }


def load_or_create_manifest(
    path: Path, df: pd.DataFrame, limit: int, seed: int, source_manifest: Path | None = None
) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if source_manifest is not None:
        manifest = create_manifest_from_source(source_manifest, limit)
    else:
        manifest = create_manifest(df, limit, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


# ============================================================ per-task run


def run_single_trajectory(
    annotation_id: str,
    confirmed_task: str,
    actions,
    action_reprs: list[str],
    client: Any,
    system_prompt: str,
    max_candidates: int,
    temperature: float,
) -> list[dict]:
    """One full pass through a task's fixed step sequence, scored against
    ground-truth history at each step (standard Mind2Web protocol -- steps are scored
    independently so one wrong step doesn't cascade into contaminating later steps'
    context). Returns step dicts without a "screenshot" field; rendering happens once,
    after a trajectory has been selected (see render_trajectory_screenshots)."""
    steps_out: list[dict] = []
    action_history: list[str] = []
    for step_idx, action in enumerate(actions):
        candidates = build_candidate_list(
            action["cleaned_html"],
            action["pos_candidates"],
            action["neg_candidates"],
            max_candidates=max_candidates,
            seed=step_seed(annotation_id, step_idx),
        )
        gt_idx = ground_truth_index(candidates)
        gt_backend_id = candidates[gt_idx - 1]["backend_node_id"] if gt_idx else None

        prompt = build_user_prompt(confirmed_task, action_history, format_candidates_for_prompt(candidates))
        raw_text = ""
        parsed = None
        error = None
        try:
            raw_text, _ = client.one_step_chat(prompt, system_msg=system_prompt, temperature=temperature)
            parsed = parse_model_response(raw_text)
        except Exception as exc:  # noqa: BLE001 - keep the pipeline moving past a single bad step
            error = f"{type(exc).__name__}: {exc}"

        pred_backend_id = None
        pred_op = ""
        pred_value = ""
        if parsed is not None:
            try:
                choice = int(parsed["candidate"])
                if 1 <= choice <= len(candidates):
                    pred_backend_id = candidates[choice - 1]["backend_node_id"]
            except (TypeError, ValueError):
                pass
            pred_op = str(parsed.get("operation", "")).upper()
            pred_value = str(parsed.get("value", ""))

        gt_op = action["operation"]["op"]
        gt_value = action["operation"]["value"]
        scorable = gt_backend_id is not None
        element_match = scorable and pred_backend_id == gt_backend_id
        op_match = pred_op == gt_op
        value_match = True
        if gt_op in ("TYPE", "SELECT"):
            value_match = pred_value.strip().lower() == str(gt_value).strip().lower()
        step_success = scorable and element_match and op_match and value_match

        step_action_repr = action_reprs[step_idx] if step_idx < len(action_reprs) else f"{gt_op} (no repr)"
        steps_out.append({
            "step": step_idx,
            "action_repr": step_action_repr,
            "candidates_shown": len(candidates),
            "scorable": scorable,
            "ground_truth": {"backend_node_id": gt_backend_id, "operation": gt_op, "value": gt_value},
            "prediction": {"backend_node_id": pred_backend_id, "operation": pred_op, "value": pred_value},
            "element_match": bool(element_match),
            "operation_match": bool(op_match),
            "value_match": bool(value_match),
            "step_success": bool(step_success),
            "reason": (parsed or {}).get("reason", ""),
            "raw_response": raw_text,
            "error": error,
        })
        # ground-truth history, not the model's own (possibly wrong) prediction --
        # standard Mind2Web protocol scores each step independently
        action_history.append(step_action_repr)
    return steps_out


def trajectory_task_success(steps_out: list[dict]) -> bool:
    scorable_steps = [s for s in steps_out if s["scorable"]]
    return bool(scorable_steps) and all(s["step_success"] for s in scorable_steps)


def render_trajectory_screenshots(
    annotation_id: str,
    actions,
    steps_out: list[dict],
    renderer: Mind2WebRenderer,
    task_dir: Path,
    multimodal_index: MultimodalIndex | None,
) -> None:
    """Fills in each step dict's "screenshot" (and possibly "error") field in place,
    for the single trajectory that was selected for this task."""
    for step_idx, step in enumerate(steps_out):
        action = actions[step_idx]
        gt_backend_id = step["ground_truth"]["backend_node_id"]
        pred_backend_id = step["prediction"]["backend_node_id"]
        screenshot_name = f"screenshot_step_{step_idx}.png"
        error = step.get("error")
        try:
            used_real = False
            if multimodal_index is not None:
                mm_step = multimodal_index.get_step(annotation_id, step_idx)
                if mm_step is not None:
                    mm_candidates = mm_step["candidates_by_id"]
                    gt_bbox = parse_bbox(mm_candidates[gt_backend_id]["attributes"]) if gt_backend_id in mm_candidates else None
                    pred_bbox = parse_bbox(mm_candidates[pred_backend_id]["attributes"]) if pred_backend_id in mm_candidates else None
                    if gt_bbox or pred_bbox:
                        highlight_and_save(
                            mm_step["screenshot_bytes"],
                            task_dir / screenshot_name,
                            gt_bbox,
                            pred_bbox,
                            bool(step["element_match"]),
                        )
                        used_real = True
            if not used_real:
                # fall back to re-rendering the (unstyled) cleaned_html when this step
                # has no matching real screenshot in Multimodal-Mind2Web
                renderer.screenshot(
                    action["cleaned_html"],
                    task_dir / screenshot_name,
                    gt_backend_id,
                    pred_backend_id,
                    bool(step["element_match"]),
                )
        except Exception as exc:  # noqa: BLE001
            error = (error + " | " if error else "") + f"render error: {type(exc).__name__}: {exc}"
        step["error"] = error
        step["screenshot"] = screenshot_name


def process_task(
    row: pd.Series,
    client: Any,
    renderer: Mind2WebRenderer,
    output_root: Path,
    max_candidates: int,
    memory_ctx: dict | None = None,
    multimodal_index: MultimodalIndex | None = None,
    scaling_k: int = 1,
    efm_ctx: dict | None = None,
) -> dict:
    annotation_id = row["annotation_id"]
    website = row["website"]
    confirmed_task = row["confirmed_task"]
    actions = row["actions"]
    action_reprs = list(row["action_reprs"])

    task_dir = output_root / f"mind2web.{annotation_id}"
    task_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = SYSTEM_PROMPT
    retrieved_memory: list[dict] = []
    memory_error = None
    if memory_ctx is not None:
        try:
            retrieved_memory = memory_lib.select_memory(
                memory_ctx["embed_client"],
                memory_ctx["memory_bank_path"],
                memory_ctx["embeddings_path"],
                confirmed_task,
                annotation_id,
                top_n=memory_ctx["top_n"],
            )
            memory_block = memory_lib.format_memory_for_prompt(retrieved_memory)
            if memory_block:
                system_prompt = SYSTEM_PROMPT + "\n\n" + memory_block
        except Exception as exc:  # noqa: BLE001 - a retrieval hiccup shouldn't kill the run
            memory_error = f"retrieval error: {type(exc).__name__}: {exc}"
            print(f"  [memory] {memory_error}", flush=True)

    efm_injected: list[tuple[str, dict]] = []
    if efm_ctx is not None:
        try:
            efm_injected = efm_retrieve(
                query=confirmed_task, now=efm_ctx["now"], task_id=annotation_id,
                items=efm_ctx["items"], experiences=efm_ctx["experiences"], hp=efm_ctx["hp"],
                query_cache_path=str(efm_ctx["query_cache_path"]),
                desc_cache_path=str(efm_ctx["desc_cache_path"]),
                n_inject=efm_ctx["n_inject"],
            )
            if efm_injected:
                mem_items = [efm_format_item(n, item) for n, (_iid, item) in enumerate(efm_injected, start=1)]
                system_prompt = SYSTEM_PROMPT + "\n\n" + memory_lib.MEM_INSTRUCTION + "\n\n" + "\n\n".join(mem_items)
            efm_store.save_json(str(efm_ctx["items_path"]), efm_ctx["items"])  # persist n_retrieved bumps
        except Exception as exc:  # noqa: BLE001 - a retrieval hiccup shouldn't kill the run
            memory_error = f"efm retrieval error: {type(exc).__name__}: {exc}"
            print(f"  [efm] {memory_error}", flush=True)

    # MaTTS parallel scaling (paper Section 3.3): k independent trajectories under the
    # same retrieved memory, at temperature 0.7 (matching the paper) so they actually
    # diverge, then an LLM Best-of-N judge picks the one to score and induce memory
    # from. At k=1 this collapses to the original single-pass, temperature-0.0 behavior.
    scaling_info = None
    if scaling_k > 1:
        trajectories = [
            run_single_trajectory(
                annotation_id, confirmed_task, actions, action_reprs,
                client, system_prompt, max_candidates, temperature=0.7,
            )
            for _ in range(scaling_k)
        ]
        best_idx, bon_raw = scaling_lib.select_best_trajectory(client.one_step_chat, confirmed_task, trajectories)
        steps_out = trajectories[best_idx]
        scaling_info = {
            "k": scaling_k,
            "selected_index": best_idx,
            "bon_raw": bon_raw,
            "trajectory_task_success": [trajectory_task_success(t) for t in trajectories],
        }
    else:
        steps_out = run_single_trajectory(
            annotation_id, confirmed_task, actions, action_reprs,
            client, system_prompt, max_candidates, temperature=0.0,
        )

    render_trajectory_screenshots(annotation_id, actions, steps_out, renderer, task_dir, multimodal_index)

    scorable_steps = [s for s in steps_out if s["scorable"]]
    task_success = bool(scorable_steps) and all(s["step_success"] for s in scorable_steps)
    result = {
        "annotation_id": annotation_id,
        "website": website,
        "confirmed_task": confirmed_task,
        "num_steps": len(steps_out),
        "num_scorable_steps": len(scorable_steps),
        "task_success": task_success,
        "memory_retrieved": [item["task_id"] for item in retrieved_memory],
        "memory_error": memory_error,
        "efm_retrieved": [iid for iid, _item in efm_injected],
        "scaling": scaling_info,
        "steps": steps_out,
    }

    if memory_ctx is not None and scorable_steps:
        try:
            # paper-matching: the induction prompt (SUCCESSFUL_SI vs FAILED_SI) is chosen by
            # an LLM-as-a-Judge self-assessment, not by Mind2Web's ground truth -- ground
            # truth (task_success) stays untouched above for the reported accuracy metrics
            judged_success, judge_raw = memory_lib.judge_success(memory_ctx["model_name"], confirmed_task, steps_out)
            induced = memory_lib.induce_memory(memory_ctx["model_name"], confirmed_task, steps_out, judged_success)
            memory_lib.append_memory_bank(memory_ctx["memory_bank_path"], annotation_id, confirmed_task, judged_success, induced)
            result["judged_success"] = judged_success
            result["judge_raw"] = judge_raw
            result["memory_induced"] = induced
        except Exception as exc:  # noqa: BLE001 - an induction hiccup shouldn't kill the run
            result["memory_error"] = f"induction error: {type(exc).__name__}: {exc}"
            print(f"  [memory] {result['memory_error']}", flush=True)

    elif efm_ctx is not None and scorable_steps:
        try:
            # Same judge as reasoningbank (memory_lib.judge_success) -- EFM only changes how
            # the induced items are stored/retrieved, not how success is judged.
            judged_success, judge_raw = memory_lib.judge_success(efm_ctx["model_name"], confirmed_task, steps_out)
            think_text = "\n\n".join(s.get("reason") or "" for s in steps_out)
            efm_record_usage(
                injected=efm_injected, think_text=think_text, now=efm_ctx["now"], client=efm_ctx["client"],
                events_path=str(efm_ctx["events_path"]), task_id=annotation_id, judged_success=judged_success,
            )
            induced_raw = memory_lib.induce_memory(efm_ctx["model_name"], confirmed_task, steps_out, judged_success)
            new_items = efm_parse_memory_items(induced_raw)
            item_ids = efm_insert_fn(
                new_items, task_id=annotation_id, now=efm_ctx["now"], items=efm_ctx["items"], hp=efm_ctx["hp"],
                desc_cache_path=str(efm_ctx["desc_cache_path"]), content_cache_path=str(efm_ctx["content_cache_path"]),
                client=efm_ctx["client"], source_label="success" if judged_success else "fail",
            )
            efm_ctx["experiences"][f"ex_{annotation_id}"] = {
                "query": confirmed_task, "trajectory_ref": annotation_id, "item_ids": item_ids,
            }
            num_forgotten = efm_forget_fn(efm_ctx["items"], now=efm_ctx["now"], hp=efm_ctx["hp"],
                                           events_path=str(efm_ctx["events_path"]))
            if num_forgotten:
                print(f"  [efm] forgot {num_forgotten} item(s) after task {annotation_id}", flush=True)
            efm_store.save_json(str(efm_ctx["items_path"]), efm_ctx["items"])
            efm_store.save_json(str(efm_ctx["experiences_path"]), efm_ctx["experiences"])
            efm_ctx["now"] += 1
            efm_store.save_json(str(efm_ctx["items_path"].parent / "now.json"), {"now": efm_ctx["now"]})
            result["judged_success"] = judged_success
            result["judge_raw"] = judge_raw
            result["memory_induced"] = induced_raw
        except Exception as exc:  # noqa: BLE001 - an induction hiccup shouldn't kill the run
            result["memory_error"] = f"efm induction error: {type(exc).__name__}: {exc}"
            print(f"  [efm] {result['memory_error']}", flush=True)

    (task_dir / "trajectory.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_task_report_html(task_dir, result)
    return result


# ============================================================ reports


def write_task_report_html(task_dir: Path, result: dict) -> None:
    if result["num_scorable_steps"] == 0:
        status = "no-gt"
    elif result["task_success"]:
        status = "success"
    else:
        status = "fail"

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>Mind2Web {esc(result['annotation_id'])}</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 18px;background:#f6f7f9;color:#17202a}",
        ".meta,.step{background:white;border:1px solid #d9dee5;border-radius:12px;padding:16px;margin:14px 0}",
        ".success{color:#08783e}.fail{color:#b42318}.no-gt{color:#667085}",
        ".pill{display:inline-block;padding:2px 8px;border-radius:100px;font-size:12px;font-weight:600}",
        ".pill.ok{background:#e6f4ea;color:#08783e}.pill.bad{background:#fbe9e7;color:#b42318}.pill.skip{background:#eef0ee;color:#667085}",
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f8fafc;padding:12px;border-radius:8px}",
        "img{display:block;max-width:100%;height:auto;border:1px solid #ccd3dc;border-radius:8px;margin-top:12px}",
        "</style></head><body>",
        f"<h1>{esc(result['website'])} &mdash; <span class='{status}'>{esc(status)}</span></h1>",
        "<section class='meta'>",
        f"<h2>Task</h2><p>{esc(result['confirmed_task'])}</p>",
        f"<p><strong>Scorable steps:</strong> {result['num_scorable_steps']}/{result['num_steps']}"
        " (steps with no surviving ground-truth candidate are shown but excluded from scoring)</p>",
    ]
    if result.get("scaling"):
        sc = result["scaling"]
        parts.append(
            f"<p><strong>MaTTS scaling:</strong> k={sc['k']}, Best-of-N selected trajectory "
            f"#{sc['selected_index'] + 1}, per-trajectory task_success={sc['trajectory_task_success']}</p>"
        )
    parts.append("</section><h2>Steps</h2>")
    for step in result["steps"]:
        if not step["scorable"]:
            pill = "<span class='pill skip'>no ground truth</span>"
        elif step["step_success"]:
            pill = "<span class='pill ok'>correct</span>"
        else:
            pill = "<span class='pill bad'>wrong</span>"
        parts.append("<section class='step'>")
        parts.append(f"<h3>Step {step['step']} {pill}</h3>")
        parts.append(f"<p><strong>Ground-truth action:</strong> {esc(step['action_repr'])}</p>")
        parts.append(
            "<p><strong>Predicted:</strong> "
            f"element={esc(step['prediction']['backend_node_id'])} "
            f"op={esc(step['prediction']['operation'])} "
            f"value={esc(step['prediction']['value'])}</p>"
        )
        if step["reason"]:
            parts.append(f"<details><summary>Model reasoning</summary><pre>{esc(step['reason'])}</pre></details>")
        if step["error"]:
            parts.append(f"<pre>error: {esc(step['error'])}</pre>")
        parts.append(f"<img src='{esc(step['screenshot'])}' alt='step {step['step']} screenshot'>")
        parts.append("</section>")
    parts.append("</body></html>")
    (task_dir / "task_report.html").write_text("".join(parts), encoding="utf-8")


def write_review_html(output_root: Path, results: list[dict], domain_label: str = "") -> None:
    title = f"Mind2Web review — {domain_label} domain" if domain_label else "Mind2Web review"
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{esc(title)}</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:24px;background:#f5f7fa;color:#17202a}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}",
        ".card{background:white;border:1px solid #d9dee5;border-radius:12px;padding:14px;overflow:hidden}",
        ".success{border-top:5px solid #12a05c}.fail{border-top:5px solid #e5484d}.no-gt{border-top:5px solid #98a2b3}",
        "img{width:100%;height:210px;object-fit:contain;background:#eef1f5;border-radius:8px}",
        "a{color:#175cd3;text-decoration:none}",
        "</style></head><body>",
        f"<h1>{esc(title)}</h1>",
        "<p>Offline replay: pages don't actually respond to clicks. Green outline = ground-truth "
        "element, green/red outline = the model's pick (green if correct, red if wrong).</p>",
        "<div class='grid'>",
    ]
    for result in results:
        if result["num_scorable_steps"] == 0:
            status = "no-gt"
        elif result["task_success"]:
            status = "success"
        else:
            status = "fail"
        task_dir_name = f"mind2web.{result['annotation_id']}"
        last_screenshot = result["steps"][-1]["screenshot"] if result["steps"] else ""
        parts.append(f"<article class='card {status}'>")
        parts.append(
            f"<h2>{esc(result['website'])}</h2>"
            f"<p><strong>{esc(status)}</strong> &middot; {result['num_scorable_steps']}/{result['num_steps']} scorable steps</p>"
        )
        parts.append(f"<p>{esc(result['confirmed_task'][:140])}</p>")
        if last_screenshot:
            parts.append(f"<img src='{esc(task_dir_name)}/{esc(last_screenshot)}' alt='last screenshot'>")
        parts.append(f"<p><a href='{esc(task_dir_name)}/task_report.html'>Open trajectory</a></p>")
        parts.append("</article>")
    parts.append("</div></body></html>")
    (output_root / "review.html").write_text("".join(parts), encoding="utf-8")


def compute_summary(results: list[dict]) -> dict:
    def rate(steps: list[dict], predicate) -> float | None:
        return sum(1 for s in steps if predicate(s)) / len(steps) if steps else None

    def task_success_rate(rows: list[dict]) -> float | None:
        scorable_rows = [r for r in rows if r["num_scorable_steps"] > 0]
        if not scorable_rows:
            return None
        return sum(1 for r in scorable_rows if r["task_success"]) / len(scorable_rows)

    all_scorable_steps = [s for r in results for s in r["steps"] if s["scorable"]]
    overall = {
        "tasks": len(results),
        "steps": sum(r["num_steps"] for r in results),
        "scorable_steps": len(all_scorable_steps),
        "element_accuracy": rate(all_scorable_steps, lambda s: s["element_match"]),
        "operation_accuracy": rate(all_scorable_steps, lambda s: s["operation_match"]),
        "step_success_rate": rate(all_scorable_steps, lambda s: s["step_success"]),
        "task_success_rate": task_success_rate(results),
    }

    by_website: dict[str, dict] = {}
    for website in sorted({r["website"] for r in results}):
        site_results = [r for r in results if r["website"] == website]
        site_scorable_steps = [s for r in site_results for s in r["steps"] if s["scorable"]]
        by_website[website] = {
            "tasks": len(site_results),
            "element_accuracy": rate(site_scorable_steps, lambda s: s["element_match"]),
            "operation_accuracy": rate(site_scorable_steps, lambda s: s["operation_match"]),
            "step_success_rate": rate(site_scorable_steps, lambda s: s["step_success"]),
            "task_success_rate": task_success_rate(site_results),
        }

    return {"schema_version": SCHEMA_VERSION, "updated_at": utc_now(), "overall": overall, "by_website": by_website}


def write_summary_md(output_root: Path, summary: dict, domain_label: str = "") -> None:
    def pct(value: float | None) -> str:
        return "-" if value is None else f"{value:.1%}"

    overall = summary["overall"]
    heading = f"# Mind2Web — {domain_label} domain" if domain_label else "# Mind2Web"
    lines = [
        heading,
        "",
        f"Tasks: {overall['tasks']} &middot; Scorable steps: {overall['scorable_steps']}/{overall['steps']}",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Element Accuracy | {pct(overall['element_accuracy'])} |",
        f"| Operation Accuracy | {pct(overall['operation_accuracy'])} |",
        f"| Step Success Rate | {pct(overall['step_success_rate'])} |",
        f"| Task Success Rate | {pct(overall['task_success_rate'])} |",
        "",
        "## By website",
        "",
        "| Website | Tasks | Element Acc | Op Acc | Step SR | Task SR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for website, stat in summary["by_website"].items():
        lines.append(
            f"| {website} | {stat['tasks']} | {pct(stat['element_accuracy'])} | "
            f"{pct(stat['operation_accuracy'])} | {pct(stat['step_success_rate'])} | {pct(stat['task_success_rate'])} |"
        )
    lines.append("")
    (output_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


# ============================================================ main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", type=Path, default=Path("data/shopping_train.parquet"))
    parser.add_argument("--output-root", type=Path, default=Path("results_shopping"))
    parser.add_argument("--manifest", type=Path, help="Reuse an explicit manifest. Default: <output-root>/manifest.json")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--task-source",
        type=Path,
        help="Slice the first --limit annotation_ids off this existing manifest.json instead of "
             "sampling fresh, so this run's tasks are a strict subset of another run's (e.g. pairing "
             "a memory-enabled run with the same tasks a no-memory baseline already used).",
    )
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--max-candidates", type=int, default=40)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Rebuild review.html/summary from existing trajectory.json files without calling the LLM.",
    )
    parser.add_argument(
        "--memory-mode",
        default="none",
        choices=["none", "reasoningbank", "efm"],
        help="none (default) = actor-only baseline. reasoningbank = retrieve top-N similar past "
             "tasks' memory before each task and induce a new memory item after it completes. "
             "efm = Edit-Free Memory Lifecycle (insertion-time Duplicate/Group/Conflict/Novel "
             "classification, retrieval-time group collapse + conflict co-injection, "
             "usage-evidence-based forgetting) -- see efm/ and root EFM_TASK.md.",
    )
    parser.add_argument("--memory-dir", type=Path, default=Path("memories_reasoningbank"))
    parser.add_argument("--memory-top-n", type=int, default=3)
    parser.add_argument(
        "--efm-n-inject",
        type=int,
        default=3,
        help="efm only: how many items to inject per task. Match this to --memory-top-n for a "
             "same-conditions comparison with the reasoningbank arm.",
    )
    EFMHParams.add_cli_args(parser)
    parser.add_argument(
        "--scaling-k",
        type=int,
        default=1,
        help="MaTTS parallel scaling factor (paper Section 3.3): run k independent "
             "trajectories per task at temperature 0.7 and use an LLM Best-of-N judge "
             "to pick the one that's scored and used for memory induction. k=1 (default) "
             "is the original single-pass, temperature-0.0 behavior.",
    )
    parser.add_argument(
        "--multimodal-dir",
        type=Path,
        default=Path("data/multimodal"),
        help="Real Multimodal-Mind2Web screenshots (see download_multimodal.py). Steps found here are "
             "rendered from the real page image instead of re-rendering the unstyled cleaned_html.",
    )
    parser.add_argument(
        "--no-real-screenshots",
        action="store_true",
        help="Always use the Playwright cleaned_html renderer, even if real screenshots are available.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    df = load_tasks(args.data_path)
    domain_label = str(df["domain"].iloc[0]) if len(df) else ""
    manifest_path = args.manifest or args.output_root / "manifest.json"
    manifest = load_or_create_manifest(manifest_path, df, args.limit, args.seed, args.task_source)
    df = df.set_index("annotation_id").loc[manifest["annotation_ids"]].reset_index()

    print(f"Domain: {domain_label}")
    print(f"Manifest: {manifest_path.resolve()}")
    print(f"Tasks: {len(df)} (seed={manifest['seed']})")
    if "source_manifest" in manifest:
        print(f"  (sliced from {manifest['source_manifest']})")
    args.output_root.mkdir(parents=True, exist_ok=True)

    memory_ctx = None
    if args.memory_mode == "reasoningbank" and not args.report_only:
        from google import genai

        domain_key = domain_label.lower()
        memory_ctx = {
            "embed_client": genai.Client(),
            "memory_bank_path": args.memory_dir / f"{domain_key}.jsonl",
            "embeddings_path": args.memory_dir / f"{domain_key}_embeddings.jsonl",
            "top_n": args.memory_top_n,
            "model_name": args.model,
        }
        print(f"Memory mode: reasoningbank (bank={memory_ctx['memory_bank_path']}, top_n={args.memory_top_n})")

    efm_ctx = None
    if args.memory_mode == "efm" and not args.report_only:
        domain_key = domain_label.lower()
        efm_dir = args.memory_dir / domain_key
        items_path = efm_dir / "items.json"
        experiences_path = efm_dir / "experiences.json"
        now_path = efm_dir / "now.json"
        efm_ctx = {
            "items": efm_store.load_json(str(items_path), {}),
            "experiences": efm_store.load_json(str(experiences_path), {}),
            "now": efm_store.load_json(str(now_path), {"now": 0})["now"],
            "hp": EFMHParams.from_args(args),
            "query_cache_path": efm_dir / "query_embeddings.jsonl",
            "desc_cache_path": efm_dir / "desc_embeddings.jsonl",
            "content_cache_path": efm_dir / "content_embeddings.jsonl",
            "events_path": efm_dir / "efm_events.jsonl",
            "items_path": items_path,
            "experiences_path": experiences_path,
            "n_inject": args.efm_n_inject,
            "model_name": args.model,
            "client": CLIENT_DICT[args.model](model_name=args.model),
        }
        print(f"Memory mode: efm (dir={efm_dir}, n_inject={args.efm_n_inject})")

    multimodal_index = None
    if not args.no_real_screenshots and not args.report_only and (args.multimodal_dir / "train").exists():
        annotation_ids = set(manifest["annotation_ids"])
        multimodal_index = MultimodalIndex(args.multimodal_dir, annotation_ids, split="train")
        print(f"Real screenshots: matched {len(multimodal_index.by_task)}/{len(annotation_ids)} tasks in {args.multimodal_dir}")

    def refresh_reports(current_results: list[dict]) -> dict:
        write_review_html(args.output_root, current_results, domain_label)
        current_summary = compute_summary(current_results)
        current_summary["domain"] = domain_label
        (args.output_root / "summary.json").write_text(
            json.dumps(current_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_summary_md(args.output_root, current_summary, domain_label)
        return current_summary

    results: list[dict] = []
    if args.report_only:
        for _, row in df.iterrows():
            trace_path = args.output_root / f"mind2web.{row['annotation_id']}" / "trajectory.json"
            if trace_path.exists():
                results.append(json.loads(trace_path.read_text(encoding="utf-8")))
    else:
        client = CLIENT_DICT[args.model](model_name=args.model)
        with Mind2WebRenderer(headless=args.headless) as renderer:
            for i, (_, row) in enumerate(df.iterrows(), 1):
                trace_path = args.output_root / f"mind2web.{row['annotation_id']}" / "trajectory.json"
                if trace_path.exists() and not args.rerun_completed:
                    print(f"[{i}/{len(df)}] {row['annotation_id']} already complete; skipping")
                    results.append(json.loads(trace_path.read_text(encoding="utf-8")))
                    continue
                print(
                    f"[{i}/{len(df)}] {row['website']} / {row['annotation_id']}: "
                    f"{row['confirmed_task'][:80]}",
                    flush=True,
                )
                started = time.monotonic()
                result = process_task(
                    row, client, renderer, args.output_root, args.max_candidates, memory_ctx, multimodal_index,
                    scaling_k=args.scaling_k, efm_ctx=efm_ctx,
                )
                elapsed = time.monotonic() - started
                scaling_note = ""
                if result.get("scaling"):
                    sc = result["scaling"]
                    scaling_note = (
                        f" scaling_k={sc['k']} selected={sc['selected_index']} "
                        f"per_trajectory_success={sc['trajectory_task_success']}"
                    )
                print(
                    f"  -> task_success={result['task_success']} "
                    f"scorable_steps={result['num_scorable_steps']}/{result['num_steps']} "
                    f"elapsed={elapsed:.1f}s{scaling_note}",
                    flush=True,
                )
                results.append(result)
                # refresh after every task, not just at the end, so review.html/summary.*
                # reflect live progress for a still-running (possibly hours-long) pass
                refresh_reports(results)

    summary = refresh_reports(results)
    print(json.dumps(summary["overall"], ensure_ascii=False))
    print(f"Reports: {args.output_root.resolve()}")
    print(f"Gallery: {(args.output_root / 'review.html').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
