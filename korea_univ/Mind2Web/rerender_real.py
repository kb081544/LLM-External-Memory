"""Retroactively replace the fake (rendered-cleaned_html) screenshots in an existing
results directory with real Multimodal-Mind2Web screenshots, highlighted the same way
(green = ground truth, green/red = predicted correct/wrong). No LLM calls -- just reads
each task's existing trajectory.json for the GT/predicted backend_node_id per step and
overwrites screenshot_step_N.png in place, so task_report.html/review.html pick up the
new images with no other changes needed.

Usage (run from Mind2Web/):
    python rerender_real.py --results-dir results_shopping_memory_gpt-5.4-mini
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_lookup import MultimodalIndex, parse_bbox
from render_real import highlight_and_save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--multimodal-dir", type=Path, default=Path("data/multimodal"))
    parser.add_argument("--split", default="train")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task_dirs = sorted(args.results_dir.glob("mind2web.*"))
    annotation_ids = {d.name.removeprefix("mind2web.") for d in task_dirs}
    print(f"Loading multimodal data for {len(annotation_ids)} tasks from {args.multimodal_dir}/{args.split}/ ...")
    index = MultimodalIndex(args.multimodal_dir, annotation_ids, split=args.split)
    print(f"Found multimodal rows for {len(index.by_task)}/{len(annotation_ids)} tasks")

    tasks_touched = 0
    steps_drawn = 0
    steps_missing = 0
    for task_dir in task_dirs:
        annotation_id = task_dir.name.removeprefix("mind2web.")
        trace_path = task_dir / "trajectory.json"
        if not trace_path.exists():
            continue
        result = json.loads(trace_path.read_text(encoding="utf-8"))
        touched_this_task = False
        for step in result["steps"]:
            mm_step = index.get_step(annotation_id, step["step"])
            if mm_step is None:
                steps_missing += 1
                continue
            candidates = mm_step["candidates_by_id"]
            gt_id = step["ground_truth"].get("backend_node_id")
            pred_id = step["prediction"].get("backend_node_id")
            gt_bbox = parse_bbox(candidates[gt_id]["attributes"]) if gt_id in candidates else None
            pred_bbox = parse_bbox(candidates[pred_id]["attributes"]) if pred_id in candidates else None
            if gt_bbox is None and pred_bbox is None:
                steps_missing += 1
                continue
            out_path = task_dir / step["screenshot"]
            drew = highlight_and_save(
                mm_step["screenshot_bytes"], out_path, gt_bbox, pred_bbox, bool(step["element_match"])
            )
            if drew:
                steps_drawn += 1
                touched_this_task = True
        if touched_this_task:
            tasks_touched += 1

    print(f"Redrew {steps_drawn} step screenshots across {tasks_touched} tasks")
    print(f"Steps with no multimodal match: {steps_missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
