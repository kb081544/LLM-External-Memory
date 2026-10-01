"""Look up real screenshots + candidate bounding boxes from the downloaded
osunlp/Multimodal-Mind2Web shards (data/multimodal/<split>/*.parquet), keyed by the
same annotation_id used in the text-only osunlp/Mind2Web dataset our pipeline already
runs against. Multimodal-Mind2Web is one row per step (not nested like the text-only
dataset), so this groups rows by annotation_id and orders them by target_action_index.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd

COLUMNS = ["annotation_id", "target_action_index", "screenshot", "pos_candidates", "neg_candidates"]


def _parse_candidate(raw: str) -> dict:
    c = json.loads(raw)
    attrs = c.get("attributes")
    if isinstance(attrs, str):
        try:
            attrs = json.loads(attrs)
        except json.JSONDecodeError:
            attrs = {}
    c["attributes"] = attrs or {}
    return c


def parse_bbox(attrs: dict) -> tuple[float, float, float, float] | None:
    """Returns (x, y, width, height) in screenshot pixel coordinates, or None."""
    raw = attrs.get("bounding_box_rect")
    if not raw:
        return None
    try:
        x, y, w, h = (float(v) for v in raw.split(","))
        return x, y, w, h
    except ValueError:
        return None


class MultimodalIndex:
    """Loads only the shards/rows needed for a given set of annotation_ids, once."""

    def __init__(self, data_dir: Path, annotation_ids: set[str], split: str = "train"):
        self.by_task: dict[str, list[dict]] = {}
        paths = sorted(glob.glob(str(data_dir / split / "*.parquet")))
        remaining = set(annotation_ids)
        for path in paths:
            if not remaining:
                break
            df = pd.read_parquet(path, columns=COLUMNS)
            hit = df[df["annotation_id"].isin(remaining)]
            if hit.empty:
                continue
            for aid, group in hit.groupby("annotation_id"):
                group = group.assign(_idx=group["target_action_index"].astype(int)).sort_values("_idx")
                self.by_task[aid] = group.to_dict("records")
                remaining.discard(aid)

    def get_step(self, annotation_id: str, step_idx: int) -> dict | None:
        rows = self.by_task.get(annotation_id)
        if not rows:
            return None
        for row in rows:
            if row["_idx"] == step_idx:
                candidates = [_parse_candidate(c) for c in row["pos_candidates"]] + \
                             [_parse_candidate(c) for c in row["neg_candidates"]]
                by_backend_id = {c["backend_node_id"]: c for c in candidates}
                return {
                    "screenshot_bytes": row["screenshot"]["bytes"],
                    "candidates_by_id": by_backend_id,
                }
        return None
