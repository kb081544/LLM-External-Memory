"""Run a reproducible, small WebArena benchmark across LLMs.

The default benchmark samples 50 shopping and 50 shopping_admin tasks.  The
sample is stored in a manifest so that later model runs use exactly the same
tasks.  Success is measured with WebArena's ground-truth cumulative reward;
execution/API failures are reported separately and never counted as task
failures.

Examples (run from WebArena/):
    python run_model_comparison.py --model gemini-3.1-flash-lite
    python run_model_comparison.py --model gemini-3.5-flash-lite
    python run_model_comparison.py --model openai/gpt-4.1-mini
    python run_model_comparison.py --model gemini-3.5-flash-lite --report-only
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html as html_lib
import json
import os
import pickle
import random
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SITES = ("shopping", "shopping_admin")
DEFAULT_COMPARISON_MODELS = (
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
)
SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def model_slug(model: str) -> str:
    """Return a filesystem-safe, still-readable model identifier."""
    return re.sub(r"[^A-Za-z0-9._-]+", "__", model).strip("._-") or "model"


def load_configs(config_dir: Path) -> dict[int, dict[str, Any]]:
    configs: dict[int, dict[str, Any]] = {}
    for path in config_dir.glob("*.json"):
        if not path.stem.isdigit():
            continue
        with path.open(encoding="utf-8") as file:
            config = json.load(file)
        task_id = int(config["task_id"])
        configs[task_id] = config
    if not configs:
        raise FileNotFoundError(
            f"No numeric task configs found in {config_dir}. "
            "Generate WebArena/config_files first."
        )
    return configs


def _site_seed(seed: int, site: str) -> int:
    digest = hashlib.sha256(f"{seed}:{site}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def create_manifest(
    configs: dict[int, dict[str, Any]],
    sites: Iterable[str],
    limit_per_site: int,
    seed: int,
) -> dict[str, Any]:
    if limit_per_site <= 0:
        raise ValueError("--limit-per-site must be greater than zero")

    selected: dict[str, list[dict[str, Any]]] = {}
    for site in sites:
        candidates = [
            config for config in configs.values() if config.get("sites") == [site]
        ]
        candidates.sort(key=lambda config: int(config["task_id"]))
        if len(candidates) < limit_per_site:
            raise ValueError(
                f"{site} has only {len(candidates)} single-site tasks, "
                f"but {limit_per_site} were requested"
            )

        rng = random.Random(_site_seed(seed, site))
        tasks = rng.sample(candidates, limit_per_site)
        tasks.sort(key=lambda config: int(config["task_id"]))
        selected[site] = [
            {
                "task_id": int(config["task_id"]),
                "intent_template_id": config.get("intent_template_id"),
            }
            for config in tasks
        ]

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "selection": "seeded_random_without_replacement",
        "seed": seed,
        "limit_per_site": limit_per_site,
        "sites": selected,
    }


def validate_manifest(
    manifest: dict[str, Any],
    configs: dict[int, dict[str, Any]],
    requested_sites: Iterable[str],
) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported manifest schema: {manifest.get('schema_version')!r}"
        )

    manifest_sites = manifest.get("sites")
    if not isinstance(manifest_sites, dict):
        raise ValueError("Manifest field 'sites' must be an object")

    for site in requested_sites:
        tasks = manifest_sites.get(site)
        if not isinstance(tasks, list) or not tasks:
            raise ValueError(f"Manifest has no tasks for {site}")
        seen: set[int] = set()
        for task in tasks:
            task_id = int(task["task_id"])
            if task_id in seen:
                raise ValueError(f"Manifest repeats task {task_id} for {site}")
            seen.add(task_id)
            config = configs.get(task_id)
            if config is None:
                raise ValueError(f"Manifest task {task_id} has no config file")
            if config.get("sites") != [site]:
                raise ValueError(
                    f"Manifest task {task_id} belongs to {config.get('sites')}, not [{site}]"
                )


def load_or_create_manifest(
    path: Path,
    configs: dict[int, dict[str, Any]],
    sites: list[str],
    limit_per_site: int,
    seed: int,
) -> dict[str, Any]:
    if path.exists():
        with path.open(encoding="utf-8") as file:
            manifest = json.load(file)
        validate_manifest(manifest, configs, sites)
        return manifest

    manifest = create_manifest(configs, sites, limit_per_site, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return manifest


def read_json(path: Path) -> Any | None:
    try:
        with path.open(encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


def reward_from_result(result_dir: Path) -> float | None:
    summary = read_json(result_dir / "summary_info.json")
    if not isinstance(summary, dict):
        return None
    # BrowserGym writes a summary even when the episode crashes.  Such a zero
    # reward is an execution error, not evidence that the agent attempted and
    # failed the task.
    if summary.get("err_msg"):
        return None
    reward = summary.get("cum_reward")
    if isinstance(reward, bool):
        return float(reward)
    if isinstance(reward, (int, float)):
        return float(reward)
    return None


def error_from_result(result_dir: Path) -> str:
    summary = read_json(result_dir / "summary_info.json")
    if not isinstance(summary, dict):
        return ""
    error = summary.get("err_msg")
    return str(error) if error else ""


def is_permanent_api_error(message: str) -> bool:
    normalized = message.upper()
    return any(
        marker in normalized
        for marker in (
            "400 INVALID_ARGUMENT",
            "401 UNAUTHENTICATED",
            "403 PERMISSION_DENIED",
            "404 NOT_FOUND",
        )
    )


def numbered_files(folder: Path, pattern: str) -> list[Path]:
    def number(path: Path) -> int:
        match = re.search(r"(\d+)", path.name)
        return int(match.group(1)) if match else -1

    return sorted(folder.glob(pattern), key=number)


def write_task_report(
    result_dir: Path,
    config: dict[str, Any],
    success_reward: float,
) -> None:
    """Export compressed BrowserGym steps as human-readable JSON and HTML."""
    if not result_dir.is_dir():
        return

    trace: list[dict[str, Any]] = []
    for step_file in numbered_files(result_dir, "step_*.pkl.gz"):
        match = re.search(r"step_(\d+)", step_file.name)
        step_number = int(match.group(1)) if match else len(trace)
        try:
            with gzip.open(step_file, "rb") as file:
                step_info = pickle.load(file)
            agent_info = getattr(step_info, "agent_info", {}) or {}
            action = getattr(step_info, "action", "") or agent_info.get("action", "")
            think = agent_info.get("think", "")
            screenshot_name = f"screenshot_step_{step_number}.png"
            screenshot = screenshot_name if (result_dir / screenshot_name).exists() else None
            trace.append(
                {
                    "step": step_number,
                    "think": str(think or ""),
                    "action": str(action or ""),
                    "screenshot": screenshot,
                }
            )
        except Exception as error:
            trace.append(
                {
                    "step": step_number,
                    "think": "",
                    "action": "",
                    "screenshot": None,
                    "read_error": f"{type(error).__name__}: {error}",
                }
            )

    reward = reward_from_result(result_dir)
    execution_error = error_from_result(result_dir)
    if reward is not None:
        status = "success" if reward >= success_reward else "fail"
    elif execution_error:
        status = "error"
    else:
        status = "pending"

    report = {
        "task_id": int(config["task_id"]),
        "intent": config.get("intent", ""),
        "sites": config.get("sites", []),
        "status": status,
        "cum_reward": reward,
        "execution_error": execution_error,
        "memory": {
            "enabled": False,
            "items": [],
            "note": (
                "run_model_comparison.py currently benchmarks the actor without "
                "ReasoningBank retrieval or memory induction."
            ),
        },
        "steps": trace,
    }
    write_json(result_dir / "trajectory.json", report)

    status_class = status if status in {"success", "fail", "error"} else "pending"
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>WebArena task {int(config['task_id'])}</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;max-width:1200px;margin:24px auto;padding:0 18px;background:#f6f7f9;color:#17202a}",
        ".meta,.step{background:white;border:1px solid #d9dee5;border-radius:12px;padding:16px;margin:14px 0}",
        ".success{color:#08783e}.fail,.error{color:#b42318}.pending{color:#667085}",
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f8fafc;padding:12px;border-radius:8px}",
        "img{display:block;max-width:100%;height:auto;border:1px solid #ccd3dc;border-radius:8px;margin-top:12px}",
        "</style></head><body>",
        f"<h1>Task {int(config['task_id'])}: <span class='{status_class}'>{html_lib.escape(status)}</span></h1>",
        "<section class='meta'>",
        f"<h2>Intent</h2><p>{html_lib.escape(str(config.get('intent', '')))}</p>",
        f"<p><strong>Reward:</strong> {html_lib.escape(str(reward))}</p>",
        "<h2>Memory</h2><p>Disabled — this model-comparison run does not retrieve or induce ReasoningBank memory.</p>",
    ]
    if execution_error:
        parts.append(f"<h2>Execution error</h2><pre>{html_lib.escape(execution_error)}</pre>")
    parts.append("</section><h2>Trajectory</h2>")
    for step in trace:
        parts.extend(
            [
                "<section class='step'>",
                f"<h3>Step {step['step']}</h3>",
                f"<h4>Action</h4><pre>{html_lib.escape(step.get('action', ''))}</pre>",
            ]
        )
        if step.get("think"):
            parts.append(
                f"<details><summary>Saved reasoning</summary><pre>{html_lib.escape(step['think'])}</pre></details>"
            )
        if step.get("screenshot"):
            screenshot = html_lib.escape(step["screenshot"], quote=True)
            parts.append(f"<img src='{screenshot}' alt='Step {step['step']} screenshot'>")
        if step.get("read_error"):
            parts.append(f"<pre>{html_lib.escape(step['read_error'])}</pre>")
        parts.append("</section>")
    parts.append("</body></html>")
    (result_dir / "task_report.html").write_text("".join(parts), encoding="utf-8")


def newest_mtime(path: Path) -> float:
    newest = 0.0
    if not path.exists():
        return newest
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        try:
            newest = max(newest, item.stat().st_mtime)
        except OSError:
            pass
    return newest


def stop_process_tree(process: subprocess.Popen[Any]) -> None:
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


def run_with_idle_timeout(
    command: list[str], results_dir: Path, idle_timeout: int
) -> tuple[int | None, float]:
    popen_kwargs = {"start_new_session": True} if os.name != "nt" else {}
    started = time.monotonic()
    process = subprocess.Popen(command, **popen_kwargs)
    last_activity = time.monotonic()
    last_mtime = newest_mtime(results_dir)

    while process.poll() is None:
        time.sleep(5)
        current_mtime = newest_mtime(results_dir)
        if current_mtime > last_mtime:
            last_mtime = current_mtime
            last_activity = time.monotonic()
        if time.monotonic() - last_activity >= idle_timeout:
            stop_process_tree(process)
            return None, time.monotonic() - started
    return process.returncode, time.monotonic() - started


def load_state(path: Path) -> dict[str, Any]:
    state = read_json(path)
    if isinstance(state, dict) and isinstance(state.get("tasks"), dict):
        return state
    return {"schema_version": SCHEMA_VERSION, "tasks": {}}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary, path)


def task_key(site: str, task_id: int) -> str:
    return f"{site}:{task_id}"


def collect_rows(
    manifest: dict[str, Any],
    sites: list[str],
    model_dir: Path,
    state: dict[str, Any],
    success_reward: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    task_state = state.get("tasks", {})
    for site in sites:
        for task in manifest["sites"][site]:
            task_id = int(task["task_id"])
            result_dir = model_dir / site / f"webarena.{task_id}"
            reward = reward_from_result(result_dir)
            if reward is not None:
                status = "success" if reward >= success_reward else "fail"
                error = ""
            else:
                saved = task_state.get(task_key(site, task_id), {})
                result_error = error_from_result(result_dir)
                status = (
                    "error"
                    if result_error or saved.get("status") == "error"
                    else "pending"
                )
                error = result_error or saved.get("reason", "")
            step_count = len(list(result_dir.glob("step_*.pkl.gz"))) if result_dir.exists() else 0
            screenshots = numbered_files(result_dir, "screenshot_step_*.png")
            task_report = result_dir / "task_report.html"
            rows.append(
                {
                    "model": state.get("model", ""),
                    "website": site,
                    "task_id": task_id,
                    "intent_template_id": task.get("intent_template_id"),
                    "status": status,
                    "cum_reward": reward,
                    "steps": step_count,
                    "screenshots": len(screenshots),
                    "last_screenshot": str(screenshots[-1].resolve()) if screenshots else "",
                    "task_report": str(task_report.resolve()) if task_report.exists() else "",
                    "error": error,
                    "result_dir": str(result_dir.resolve()),
                }
            )
    return rows


def write_review_html(model_dir: Path, model: str, rows: list[dict[str, Any]]) -> None:
    """Create a screenshot-first index for reviewing every sampled task."""
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>WebArena review — {html_lib.escape(model)}</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:24px;background:#f5f7fa;color:#17202a}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}",
        ".card{background:white;border:1px solid #d9dee5;border-radius:12px;padding:14px;overflow:hidden}",
        ".success{border-top:5px solid #12a05c}.fail{border-top:5px solid #e5484d}.error{border-top:5px solid #f79009}.pending{border-top:5px solid #98a2b3}",
        "img{width:100%;height:210px;object-fit:contain;background:#eef1f5;border-radius:8px}",
        "a{color:#175cd3;text-decoration:none}code{overflow-wrap:anywhere}",
        "</style></head><body>",
        f"<h1>WebArena review — {html_lib.escape(model)}</h1>",
        "<p>Memory: <strong>disabled</strong> for this actor-only comparison run. "
        "Success/fail comes from WebArena <code>cum_reward</code>.</p>",
        "<div class='grid'>",
    ]
    for row in rows:
        status = str(row["status"])
        parts.append(f"<article class='card {html_lib.escape(status)}'>")
        parts.append(
            f"<h2>{html_lib.escape(str(row['website']))} / task {row['task_id']}</h2>"
            f"<p><strong>{html_lib.escape(status)}</strong> · reward "
            f"{html_lib.escape(str(row['cum_reward']))} · {row['steps']} steps · "
            f"{row['screenshots']} screenshots</p>"
        )
        screenshot_path = row.get("last_screenshot")
        if screenshot_path:
            relative = os.path.relpath(screenshot_path, model_dir).replace(os.sep, "/")
            parts.append(
                f"<img src='{html_lib.escape(relative, quote=True)}' "
                f"alt='Task {row['task_id']} last screenshot'>"
            )
        report_path = row.get("task_report")
        if report_path:
            relative = os.path.relpath(report_path, model_dir).replace(os.sep, "/")
            parts.append(
                f"<p><a href='{html_lib.escape(relative, quote=True)}'>Open actions and screenshots</a></p>"
            )
        elif row.get("error"):
            parts.append(f"<p>{html_lib.escape(str(row['error'])[:300])}</p>")
        parts.append("</article>")
    parts.append("</div></body></html>")
    (model_dir / "review.html").write_text("".join(parts), encoding="utf-8")


def stats_for(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        name: sum(row["status"] == name for row in rows)
        for name in ("success", "fail", "error", "pending")
    }
    completed = counts["success"] + counts["fail"]
    return {
        "planned": len(rows),
        "completed": completed,
        **counts,
        "success_rate": counts["success"] / completed if completed else None,
        "completion_rate": completed / len(rows) if rows else None,
    }


def write_reports(
    manifest_path: Path,
    manifest: dict[str, Any],
    sites: list[str],
    model: str,
    model_dir: Path,
    state: dict[str, Any],
    success_reward: float,
) -> dict[str, Any]:
    rows = collect_rows(manifest, sites, model_dir, state, success_reward)
    by_site = {
        site: stats_for([row for row in rows if row["website"] == site])
        for site in sites
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": utc_now(),
        "model": model,
        "memory_mode": "disabled",
        "success_definition": f"cum_reward >= {success_reward:g}",
        "manifest": str(manifest_path.resolve()),
        "model_preflight": state.get("model_preflight"),
        "overall": stats_for(rows),
        "by_website": by_site,
    }
    write_json(model_dir / "summary.json", summary)

    csv_path = model_dir / "task_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    write_review_html(model_dir, model, rows)

    def percent(value: float | None) -> str:
        return "-" if value is None else f"{value:.1%}"

    lines = [
        f"# WebArena model comparison: {model}",
        "",
        f"Success criterion: `{summary['success_definition']}`",
        "",
        "| Website | Planned | Completed | Success | Fail | Error | Pending | Success rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for site in sites:
        stat = by_site[site]
        lines.append(
            f"| {site} | {stat['planned']} | {stat['completed']} | "
            f"{stat['success']} | {stat['fail']} | {stat['error']} | "
            f"{stat['pending']} | {percent(stat['success_rate'])} |"
        )
    overall = summary["overall"]
    lines.append(
        f"| **Overall** | **{overall['planned']}** | **{overall['completed']}** | "
        f"**{overall['success']}** | **{overall['fail']}** | "
        f"**{overall['error']}** | **{overall['pending']}** | "
        f"**{percent(overall['success_rate'])}** |"
    )
    lines.extend(
        [
            "",
            "`error` means the task produced no valid summary and is excluded from the success rate.",
            "",
        ]
    )
    (model_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def write_comparison_reports(
    output_root: Path,
    manifest_path: Path,
    sites: list[str],
    expected_models: list[str],
) -> None:
    """Combine model summaries and keep not-yet-run comparison models visible."""
    manifest_resolved = str(manifest_path.resolve())
    rows_by_model: dict[str, dict[str, Any]] = {}
    for summary_path in (output_root / "models").glob("*/*/summary.json"):
        summary = read_json(summary_path)
        if not isinstance(summary, dict) or summary.get("manifest") != manifest_resolved:
            continue
        overall = summary.get("overall", {})
        model = str(summary.get("model", ""))
        row = {
            "model": model,
            "availability": (summary.get("model_preflight") or {}).get(
                "status", "not_checked"
            ),
            "planned": overall.get("planned"),
            "completed": overall.get("completed"),
            "success": overall.get("success"),
            "fail": overall.get("fail"),
            "error": overall.get("error"),
            "pending": overall.get("pending"),
            "success_rate": overall.get("success_rate"),
        }
        for site in sites:
            row[f"{site}_success_rate"] = summary.get("by_website", {}).get(site, {}).get(
                "success_rate"
            )
        rows_by_model[model] = row

    planned = sum(
        len(tasks)
        for site, tasks in (read_json(manifest_path) or {}).get("sites", {}).items()
        if site in sites
    )
    for model in expected_models:
        if model in rows_by_model:
            continue
        row = {
            "model": model,
            "availability": "not_checked",
            "planned": planned,
            "completed": 0,
            "success": 0,
            "fail": 0,
            "error": 0,
            "pending": planned,
            "success_rate": None,
        }
        for site in sites:
            row[f"{site}_success_rate"] = None
        rows_by_model[model] = row

    expected_order = {model: index for index, model in enumerate(expected_models)}
    comparison_rows = sorted(
        rows_by_model.values(),
        key=lambda row: (expected_order.get(row["model"], len(expected_order)), row["model"]),
    )

    csv_path = output_root / f"comparison_{manifest_path.stem}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        if comparison_rows:
            writer = csv.DictWriter(file, fieldnames=list(comparison_rows[0]))
            writer.writeheader()
            writer.writerows(comparison_rows)

    def percent(value: float | None) -> str:
        return "-" if value is None else f"{value:.1%}"

    headers = [
        "Model",
        "Availability",
        "Completed",
        "Success",
        "Fail",
        "Error",
        "Overall",
    ] + [site for site in sites]
    lines = [
        f"# Model comparison: {manifest_path.stem}",
        "",
        "| " + " | ".join(headers) + " |",
        "|---|---|---:|---:|---:|---:|---:" + "|---:" * len(sites) + "|",
    ]
    for row in comparison_rows:
        values = [
            str(row["model"]),
            str(row["availability"]),
            f"{row['completed']}/{row['planned']}",
            str(row["success"]),
            str(row["fail"]),
            str(row["error"]),
            percent(row["success_rate"]),
        ] + [percent(row[f"{site}_success_rate"]) for site in sites]
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    (output_root / f"comparison_{manifest_path.stem}.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def credential_errors(model: str) -> list[str]:
    errors: list[str] = []
    if model.startswith("gemini"):
        if not os.getenv("GOOGLE_API_KEY"):
            errors.append("GOOGLE_API_KEY is required for the Gemini Developer API")
        if os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in {"1", "true", "yes"}:
            errors.append(
                "GOOGLE_GENAI_USE_VERTEXAI is enabled; remove it to use the free Gemini Developer API"
            )
    elif model.startswith("openai/"):
        if not os.getenv("OPENAI_API_KEY"):
            errors.append("OPENAI_API_KEY is required for an openai/* actor model")
    elif model.startswith("claude"):
        for name in ("GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"):
            if not os.getenv(name):
                errors.append(f"{name} is required by this repo's AnthropicVertex client")
    else:
        errors.append(
            "Unsupported actor model prefix. Use gemini*, openai/<model>, or claude*."
        )
    return errors


def check_model_access(model: str) -> tuple[bool, str]:
    """Make one real generation request before launching an expensive browser task."""
    if not model.startswith("gemini"):
        return True, "preflight is currently implemented for Gemini models only"

    try:
        from google import genai
        from google.genai.types import GenerateContentConfig

        client = genai.Client()
        response = client.models.generate_content(
            model=model,
            contents="Reply with exactly OK.",
            config=GenerateContentConfig(
                max_output_tokens=8,
            ),
        )
        if not response.text:
            return False, "generateContent returned no text"
        return True, "generateContent succeeded"
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"


def environment_warnings(sites: Iterable[str]) -> list[str]:
    warnings: list[str] = []
    required = {"WA_HOMEPAGE"}
    if "shopping" in sites:
        required.add("WA_SHOPPING")
    if "shopping_admin" in sites:
        required.add("WA_SHOPPING_ADMIN")
    for name in sorted(required):
        bare_name = name.removeprefix("WA_")
        if not os.getenv(name) and not os.getenv(bare_name):
            warnings.append(f"{name} is not set")
    if not os.getenv("OPENAI_API_KEY"):
        warnings.append(
            "OPENAI_API_KEY is not set. Some WebArena ground-truth evaluators call an "
            "OpenAI-compatible endpoint; use OPENAI_BASE_URL with a local server if needed."
        )
    return warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the same reproducible sample of shopping and shopping_admin tasks "
            "and report GT success/fail counts."
        )
    )
    parser.add_argument(
        "--model",
        default="gemini-3.5-flash-lite",
        help=(
            "Actor model ID passed through to the provider. Current comparison "
            "defaults are gemini-3.1-flash-lite and gemini-3.5-flash-lite."
        ),
    )
    parser.add_argument(
        "--comparison-models",
        nargs="+",
        default=list(DEFAULT_COMPARISON_MODELS),
        help=(
            "Models expected in the combined comparison table. Missing runs are "
            "shown as pending."
        ),
    )
    parser.add_argument("--sites", nargs="+", choices=DEFAULT_SITES, default=list(DEFAULT_SITES))
    parser.add_argument("--limit-per-site", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config-dir", type=Path, default=Path("config_files"))
    parser.add_argument("--output-root", type=Path, default=Path("results_model_comparison"))
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Reuse an explicit manifest. Default: <output-root>/manifests/<sample>.json",
    )
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument(
        "--slow-mo",
        type=int,
        default=500,
        help="Playwright action delay in milliseconds; 500 is visible to a human observer.",
    )
    parser.add_argument("--idle-timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--pause-between-tasks", type=float, default=0.0)
    parser.add_argument(
        "--max-tasks-this-run",
        type=int,
        help="Stop after starting this many unfinished tasks; rerun the command to resume.",
    )
    parser.add_argument("--success-reward", type=float, default=1.0)
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Hide the browser window. The default is visible browser mode.",
    )
    parser.add_argument(
        "--record-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save Playwright video in addition to the per-step screenshots.",
    )
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-model-preflight",
        action="store_true",
        help="Skip the one-request generateContent availability check.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    configs = load_configs(args.config_dir)
    sample_name = (
        f"{'_'.join(args.sites)}_{args.limit_per_site}_each_seed_{args.seed}.json"
    )
    manifest_path = args.manifest or args.output_root / "manifests" / sample_name
    manifest = load_or_create_manifest(
        manifest_path, configs, args.sites, args.limit_per_site, args.seed
    )

    actual_counts = {site: len(manifest["sites"][site]) for site in args.sites}
    print(f"Manifest: {manifest_path.resolve()}")
    print("Selected tasks: " + ", ".join(f"{site}={count}" for site, count in actual_counts.items()))

    model_dir = args.output_root / "models" / model_slug(args.model) / manifest_path.stem
    state_path = model_dir / "run_state.json"
    state = load_state(state_path)
    state.update(
        {
            "schema_version": SCHEMA_VERSION,
            "model": args.model,
            "manifest": str(manifest_path.resolve()),
            "updated_at": utc_now(),
        }
    )
    write_json(state_path, state)

    def refresh_reports() -> dict[str, Any]:
        summary = write_reports(
            manifest_path,
            manifest,
            args.sites,
            args.model,
            model_dir,
            state,
            args.success_reward,
        )
        write_comparison_reports(
            args.output_root,
            manifest_path,
            args.sites,
            args.comparison_models,
        )
        return summary

    def build_existing_task_reports() -> None:
        for site in args.sites:
            for task in manifest["sites"][site]:
                task_id = int(task["task_id"])
                write_task_report(
                    model_dir / site / f"webarena.{task_id}",
                    configs[task_id],
                    args.success_reward,
                )

    if args.dry_run:
        for site in args.sites:
            ids = [str(task["task_id"]) for task in manifest["sites"][site]]
            print(f"{site}: {', '.join(ids)}")
        refresh_reports()
        print(f"Dry run only. Reports: {model_dir.resolve()}")
        return 0

    if args.report_only:
        build_existing_task_reports()
        summary = refresh_reports()
        print(json.dumps(summary["overall"], ensure_ascii=False))
        print(f"Reports: {model_dir.resolve()}")
        return 0

    preflight_errors = credential_errors(args.model)
    if preflight_errors:
        for message in preflight_errors:
            print(f"[preflight error] {message}", file=sys.stderr)
        return 2
    for message in environment_warnings(args.sites):
        print(f"[preflight warning] {message}", file=sys.stderr)

    if not args.skip_model_preflight:
        print(f"[model preflight] checking {args.model} with generateContent...", flush=True)
        available, detail = check_model_access(args.model)
        state["model_preflight"] = {
            "status": "available" if available else "unavailable",
            "detail": detail,
            "checked_at": utc_now(),
        }
        state["updated_at"] = utc_now()
        write_json(state_path, state)
        refresh_reports()
        if not available:
            print(
                f"[preflight error] {args.model} cannot generate content: {detail}",
                file=sys.stderr,
            )
            print(
                "No WebArena tasks were started. Choose a model returned by your "
                "Gemini API project.",
                file=sys.stderr,
            )
            return 2
        print(f"[model preflight] {detail}", flush=True)

    tasks_started = 0
    stop_requested = False
    for site in args.sites:
        site_output = model_dir / site
        site_output.mkdir(parents=True, exist_ok=True)
        for task in manifest["sites"][site]:
            task_id = int(task["task_id"])
            result_dir = site_output / f"webarena.{task_id}"
            if result_dir.exists() and not (result_dir / "task_report.html").exists():
                write_task_report(result_dir, configs[task_id], args.success_reward)
            if reward_from_result(result_dir) is not None and not args.rerun_completed:
                print(f"[{site} {task_id}] already complete; skipping")
                continue
            if args.max_tasks_this_run is not None and tasks_started >= args.max_tasks_this_run:
                stop_requested = True
                break

            command = [
                sys.executable,
                "run.py",
                "--task_name",
                f"webarena.{task_id}",
                "--model_name",
                args.model,
                "--results_path",
                str(site_output),
                "--max_steps",
                str(args.max_steps),
                "--slow_mo",
                str(args.slow_mo),
                "--headless",
                str(args.headless),
                "--record_video",
                str(args.record_video),
            ]
            tasks_started += 1
            last_reason = "unknown error"
            completed = False
            for attempt in range(1, args.retries + 2):
                print(
                    f"[{site} {task_id}] attempt {attempt}/{args.retries + 1}: "
                    f"{' '.join(command)}",
                    flush=True,
                )
                returncode, elapsed = run_with_idle_timeout(
                    command, site_output, args.idle_timeout
                )
                write_task_report(result_dir, configs[task_id], args.success_reward)
                reward = reward_from_result(result_dir)
                if returncode == 0 and reward is not None:
                    status = "success" if reward >= args.success_reward else "fail"
                    print(
                        f"[{site} {task_id}] {status}: reward={reward:g}, "
                        f"elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                    state["tasks"][task_key(site, task_id)] = {
                        "status": "complete",
                        "reward": reward,
                        "attempt": attempt,
                        "updated_at": utc_now(),
                    }
                    completed = True
                    break
                last_reason = (
                    f"no result activity for {args.idle_timeout}s"
                    if returncode is None
                    else f"exit code {returncode}"
                )
                if returncode == 0 and reward is None:
                    result_error = error_from_result(result_dir)
                    last_reason = result_error or (
                        "run exited successfully but summary_info.json has no numeric cum_reward"
                    )
                print(f"[{site} {task_id}] {last_reason}", file=sys.stderr, flush=True)
                if is_permanent_api_error(last_reason):
                    print(
                        f"[{site} {task_id}] permanent API error; not retrying this task",
                        file=sys.stderr,
                        flush=True,
                    )
                    break

            if not completed:
                state["tasks"][task_key(site, task_id)] = {
                    "status": "error",
                    "reason": last_reason,
                    "updated_at": utc_now(),
                }
            state["updated_at"] = utc_now()
            write_json(state_path, state)
            summary = refresh_reports()
            overall = summary["overall"]
            print(
                "[progress] "
                f"completed={overall['completed']}/{overall['planned']} "
                f"success={overall['success']} fail={overall['fail']} "
                f"error={overall['error']}",
                flush=True,
            )
            if not completed and args.fail_fast:
                stop_requested = True
                break
            if args.pause_between_tasks > 0:
                time.sleep(args.pause_between_tasks)
        if stop_requested:
            break

    summary = refresh_reports()
    overall = summary["overall"]
    print(json.dumps(overall, ensure_ascii=False))
    print(f"Reports: {model_dir.resolve()}")
    print(f"Visual review: {(model_dir / 'review.html').resolve()}")
    return 1 if overall["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
