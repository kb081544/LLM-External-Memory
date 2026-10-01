"""Build a bounded, LLM-promptable candidate shortlist for a single Mind2Web step.

Mind2Web ships up to ~2700 negative candidates per step -- far too many to put in a
prompt. The official MindAct baseline narrows this down with a trained cross-encoder
ranker before asking a generative model to pick one. No such ranker is available here,
so this module substitutes a cheap heuristic: always keep the ground-truth candidate
(when present), prefer clickable negatives, and cap the total list size. This is a
known simplification -- see the pipeline plan for the rationale.
"""

from __future__ import annotations

import json
import random
from typing import Any

from bs4 import BeautifulSoup


def _parse_attributes(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _describe_node(soup: BeautifulSoup, backend_node_id: str) -> dict:
    """Look up a backend_node_id in the step's cleaned_html and extract its text."""
    el = soup.find(attrs={"backend_node_id": backend_node_id})
    if el is None:
        return {"tag": "", "text": ""}
    text = el.get_text(" ", strip=True)
    if len(text) > 200:
        text = text[:200] + "..."
    return {"tag": el.name, "text": text}


def build_candidate_list(
    cleaned_html: str,
    pos_candidates: Any,
    neg_candidates: Any,
    max_candidates: int = 40,
    seed: int = 0,
) -> list[dict]:
    """Return a shuffled list of {backend_node_id, tag, text, attrs, is_ground_truth}.

    Always includes the (single, when present) ground-truth candidate; fills the rest
    with up to max_candidates-1 negatives, preferring clickable elements, then shuffles
    the whole list so the model can't shortcut by position.
    """
    soup = BeautifulSoup(cleaned_html, "lxml")
    rng = random.Random(seed)

    candidates: list[dict] = []
    gt_id = None
    if len(pos_candidates) > 0:
        # a step can have more than one pos_candidate; prefer the one flagged as the
        # actual annotated target over incidental ancestor/descendant matches
        pos = next((p for p in pos_candidates if p.get("is_original_target")), pos_candidates[0])
        gt_id = str(pos["backend_node_id"])
        info = _describe_node(soup, gt_id)
        candidates.append({
            "backend_node_id": gt_id,
            "tag": info["tag"] or pos.get("tag", ""),
            "text": info["text"],
            "attrs": _parse_attributes(pos["attributes"]),
            "is_ground_truth": True,
        })

    neg_pool = list(neg_candidates)
    rng.shuffle(neg_pool)

    def is_clickable(neg: Any) -> bool:
        return _parse_attributes(neg["attributes"]).get("is_clickable") == "true"

    ordered_negs = [n for n in neg_pool if is_clickable(n)] + [n for n in neg_pool if not is_clickable(n)]

    seen_ids = {gt_id} if gt_id else set()
    slots_left = max(max_candidates - len(candidates), 0)
    for neg in ordered_negs:
        if slots_left <= 0:
            break
        neg_id = str(neg["backend_node_id"])
        if neg_id in seen_ids:
            continue
        seen_ids.add(neg_id)
        info = _describe_node(soup, neg_id)
        candidates.append({
            "backend_node_id": neg_id,
            "tag": info["tag"] or neg.get("tag", ""),
            "text": info["text"],
            "attrs": _parse_attributes(neg["attributes"]),
            "is_ground_truth": False,
        })
        slots_left -= 1

    rng.shuffle(candidates)
    return candidates


def format_candidates_for_prompt(candidates: list[dict]) -> str:
    lines = []
    for i, c in enumerate(candidates, 1):
        text = c["text"] or "(no visible text)"
        attrs = c["attrs"]
        extras = ""
        if attrs.get("class"):
            extras += f' class="{attrs["class"]}"'
        if attrs.get("title"):
            extras += f' title="{attrs["title"]}"'
        lines.append(f"{i}. <{c['tag']}{extras}> {text}")
    return "\n".join(lines)


def ground_truth_index(candidates: list[dict]) -> int | None:
    for i, c in enumerate(candidates, 1):
        if c["is_ground_truth"]:
            return i
    return None
