"""Re-evaluate fuzzy_match tasks using MemEvol's more lenient llm_fuzzy_match prompt.

MemEvol changes the grading instruction from "semantically equivalent" to a
containment-based check, and redefines N/A handling. This script monkey-patches
that prompt into the installed webarena package before evaluating, so the
package itself is not modified.

Usage:
    cd ~/projects/reasoning-bank/WebArena
    python reeval_memevol_prompt.py \
        --results_dir ~/results-20260404-rb \
        --output_path ~/reeval_memevol_results.json \
        [--dry_run]   # skip LLM calls, useful for testing extraction
"""

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# 0. Set dummy webarena env vars so the package import doesn't assert-fail.
#    The actual URLs are not needed because we only call LLM judge functions,
#    not any browser/HTTP helpers.
# ---------------------------------------------------------------------------
for _var in ("REDDIT", "SHOPPING", "SHOPPING_ADMIN", "GITLAB", "WIKIPEDIA", "MAP", "HOMEPAGE"):
    if not os.environ.get(_var):
        os.environ[_var] = "http://dummy"

# ---------------------------------------------------------------------------
# 1. Apply MemEvol monkey-patch (shared with run.py via webarena_patch.py)
# ---------------------------------------------------------------------------
import webarena_patch  # noqa: F401

import webarena.evaluation_harness.helper_functions as _hf
from webarena_patch import _memevol_llm_fuzzy_match

# ---------------------------------------------------------------------------
# 2. Helper: extract final send_msg_to_user answer from experiment.log
# ---------------------------------------------------------------------------

def extract_final_response(log_path: str) -> Optional[str]:
    """Return the unquoted argument of the last send_msg_to_user() call."""
    text = Path(log_path).read_text(errors="replace")
    # Find all send_msg_to_user(...) occurrences; the arg may be on the same line
    # or immediately after an "action:" line. We scan line-by-line to handle
    # the "action:\nsend_msg_to_user(...)" pattern.
    lines = text.splitlines()
    last_call: Optional[str] = None
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "send_msg_to_user(" in line:
            last_call = line
        i += 1

    if last_call is None:
        return None

    # Extract argument: everything between the first ( and the matching last )
    start = last_call.index("send_msg_to_user(") + len("send_msg_to_user(")
    raw = last_call[start:]
    # Find the matching closing paren (simple: last ')' on the line)
    end = raw.rfind(")")
    if end == -1:
        return None
    raw_arg = raw[:end].strip()

    # Evaluate quoted string safely
    try:
        return ast.literal_eval(raw_arg)
    except Exception:
        # Fallback: strip outer quotes manually
        if (raw_arg.startswith('"') and raw_arg.endswith('"')) or \
           (raw_arg.startswith("'") and raw_arg.endswith("'")):
            return raw_arg[1:-1]
        return raw_arg


# ---------------------------------------------------------------------------
# 3. Scoring logic mirroring StringEvaluator for fuzzy_match-only configs
# ---------------------------------------------------------------------------

from webarena.evaluation_harness.evaluators import StringEvaluator
from webarena.evaluation_harness.helper_functions import llm_ua_match


def compute_new_score(pred_raw: str, config: dict) -> float:
    """Compute score using patched llm_fuzzy_match (MemEvol prompt)."""
    pred = StringEvaluator.clean_answer(pred_raw)
    ref_answers = config["eval"]["reference_answers"]
    score = 1.0

    for approach, value in ref_answers.items():
        if approach == "fuzzy_match":
            intent = config["intent"]
            if value == "N/A":
                # exact match first
                score *= StringEvaluator.exact_match(ref="N/A", pred=pred)
                if score != 1.0:
                    score = 1.0 * llm_ua_match(
                        pred=pred,
                        reference=config["eval"].get("string_note", ""),
                        question=intent,
                    )
            else:
                assert isinstance(value, list)
                for reference in value:
                    # This now calls the monkey-patched version
                    score *= _memevol_llm_fuzzy_match(pred, reference, intent)
        elif approach == "must_include":
            assert isinstance(value, list)
            for must_value in value:
                score *= StringEvaluator.must_include(
                    ref=must_value, pred=pred, tokenize=(len(value) == 1)
                )
        elif approach == "exact_match":
            score *= StringEvaluator.exact_match(ref=value, pred=pred)

    return score


# ---------------------------------------------------------------------------
# 4. Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=os.path.expanduser("~/results-20260404-rb"))
    parser.add_argument("--config_dir", default="config_files")
    parser.add_argument("--output_path", default=os.path.expanduser("~/reeval_memevol_results.json"))
    parser.add_argument("--dry_run", action="store_true",
                        help="Skip LLM calls; just verify extraction and print what would be evaluated.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    config_dir = Path(args.config_dir)
    output_path = Path(args.output_path)

    rows = []
    changed = 0
    total_fuzzy = 0

    task_dirs = sorted(results_dir.glob("webarena.*"),
                       key=lambda p: int(p.name.split(".")[1]))

    for task_dir in task_dirs:
        tid = int(task_dir.name.split(".")[1])
        config_path = config_dir / f"{tid}.json"
        if not config_path.exists():
            print(f"[SKIP] {tid}: no config")
            continue

        config = json.loads(config_path.read_text())
        eval_types = config["eval"].get("eval_types", [])
        ref_answers = config["eval"]["reference_answers"]

        # Only re-evaluate tasks that use fuzzy_match
        if not ref_answers or "fuzzy_match" not in ref_answers:
            continue

        total_fuzzy += 1

        # Get old score
        summary_path = task_dir / "summary_info.json"
        old_score = json.loads(summary_path.read_text())["cum_reward"] if summary_path.exists() else None

        # Extract final response
        log_path = task_dir / "experiment.log"
        response = extract_final_response(str(log_path)) if log_path.exists() else None

        if response is None:
            # Agent never called send_msg_to_user; use "" as pred, same as
            # the original evaluator. MemEvol's lenient prompt may still score
            # this as correct if the task context warrants it (e.g. action-only tasks).
            print(f"[INFO] {tid}: no send_msg_to_user found, using empty response")
            response = ""

        if args.dry_run:
            print(f"[DRY] {tid}: response={repr(response[:80])}  ref={ref_answers['fuzzy_match']}")
            continue

        # Compute new score
        try:
            new_score = compute_new_score(response, config)
        except Exception as e:
            print(f"[ERROR] {tid}: {e}")
            rows.append({"tid": tid, "old_score": old_score, "new_score": None,
                         "response": response, "reference": ref_answers,
                         "error": str(e)})
            continue

        delta = (new_score - old_score) if old_score is not None else None
        status = ""
        if delta and abs(delta) > 0.01:
            changed += 1
            status = f"  *** CHANGED: {old_score:.2f} -> {new_score:.2f} ***"

        print(f"[{tid:4d}] old={old_score:.2f}  new={new_score:.2f}{status}")
        rows.append({
            "tid": tid,
            "intent": config["intent"],
            "old_score": old_score,
            "new_score": new_score,
            "delta": delta,
            "response": response,
            "reference": ref_answers,
        })

    if not args.dry_run:
        output_path.write_text(json.dumps(rows, indent=2))
        total_old = sum(r["old_score"] for r in rows if r["old_score"] is not None)
        total_new = sum(r["new_score"] for r in rows if r["new_score"] is not None)
        print(f"\n{'='*60}")
        print(f"Fuzzy-match tasks evaluated: {total_fuzzy}")
        print(f"Tasks with changed score:    {changed}")
        print(f"Sum of old scores (fuzzy):   {total_old:.1f}")
        print(f"Sum of new scores (fuzzy):   {total_new:.1f}")
        print(f"Results saved to:            {output_path}")


if __name__ == "__main__":
    main()
