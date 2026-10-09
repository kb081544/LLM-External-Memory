"""ReasoningBank memory retrieval + induction for Mind2Web, adapted from
WebArena/memory_management.py (retrieval) and WebArena/induce_memory.py (induction,
reasoningbank mode only).

Differences from the WebArena version, and why:
  - Retrieval/induction happens once per TASK (keyed on `confirmed_task`), not per site --
    Mind2Web has no live episode/step loop to attach a memory_path to; the closest analogue
    is "one query per task", exactly like WebArena's per-episode memory_path.

Matches the paper (Appendix A.1, p.20 / A.2, p.21): the correctness signal fed to memory
induction (which decides SUCCESSFUL_SI vs FAILED_SI) comes from an LLM-as-a-Judge
self-assessment, NOT ground truth -- "REASONING BANK distills ... memory items from both
successful and failed experiences judged by the agent itself without ground-truth labels"
(abstract). This is deliberate: it keeps the method usable in settings with no ground truth.
Mind2Web's real ground-truth labels (element/operation/value match) are still used for the
reported accuracy metrics in run.py -- judge_success() below is *only* consulted for the
induction prompt choice, mirroring the paper's separation of "evaluation" from "what the
agent believes about its own trajectory."
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from google import genai
from google.genai.types import EmbedContentConfig

from memory_instruction import (
    SUCCESSFUL_SI, FAILED_SI, MEM_INSTRUCTION, AWM_INSTRUCTION, AWM_EXAMPLE,
    ACE_REFLECTOR_SI, ACE_REFLECTOR_FI, REME_SUCCESS_SI, REME_FAILURE_SI, MEMP_BUILD_SI,
    CER_SKILLS_DISTILL_SI, CER_REPLAY_INSTRUCTION,
)
from utils.clients import CLIENT_DICT

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 3072


def l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(x, p=2, dim=dim)


def embed_query(client: genai.Client, query: str, dimensionality: int = EMBED_DIM) -> torch.Tensor:
    resp = client.models.embed_content(
        model=EMBED_MODEL,
        contents=[query],
        config=EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=dimensionality),
    )
    return torch.tensor([resp.embeddings[0].values], dtype=torch.float32)


def load_cached_embeddings(path: Path) -> tuple[list[str], torch.Tensor]:
    ids: list[str] = []
    vecs: list[list[float]] = []
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return ids, torch.empty(0)
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            ids.append(obj["id"])
            vecs.append(obj["embedding"])
    if not vecs:
        return ids, torch.empty(0)
    emb = torch.tensor(vecs, dtype=torch.float32)
    return ids, l2_normalize(emb, dim=1)


def load_memory_bank(path: Path) -> list[dict]:
    if not path.exists():
        return []
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def select_memory(
    client: genai.Client,
    memory_bank_path: Path,
    embeddings_path: Path,
    cur_query: str,
    task_id: str,
    top_n: int = 3,
) -> list[dict]:
    """Embed cur_query, append it to the embeddings cache (side effect, matching
    WebArena's screening()), then return the top_n most similar prior memory-bank
    entries by cosine similarity. Returns [] on a cold/empty bank."""
    ids, cache_emb = load_cached_embeddings(embeddings_path)
    q_vec = embed_query(client, cur_query)

    with embeddings_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"id": task_id, "text": cur_query, "embedding": q_vec.squeeze(0).tolist()}) + "\n")

    if len(cache_emb) == 0:
        return []

    q_norm = l2_normalize(q_vec, dim=1)
    scores = (q_norm @ cache_emb.T).squeeze(0) * 100.0
    ranked = sorted(zip(ids, scores.tolist()), key=lambda x: x[1], reverse=True)
    top_ids = {tid for tid, _ in ranked[:top_n]}

    bank = load_memory_bank(memory_bank_path)
    by_id = {item["task_id"]: item for item in bank}
    out = []
    for tid, _ in ranked[:top_n]:
        if tid in by_id:
            out.append(by_id[tid])
    return out


def format_memory_for_prompt(memory_items: list[dict]) -> str:
    """Flatten retrieved memory-bank rows' memory_items into a numbered "# Memory Item i"
    block list, renumbered for this prompt, matching the format the induction prompt
    itself produces (and what tools/make_report.py reconstructs for WebArena)."""
    blocks = []
    for row in memory_items:
        for raw in row.get("memory_items", []):
            raw = raw.strip()
            if raw:
                blocks.append(raw)
    if not blocks:
        return ""
    renumbered = []
    for i, block in enumerate(blocks, 1):
        # each block starts with "# Memory Item <n>" -- replace the number only
        lines = block.splitlines()
        if lines and lines[0].strip().startswith("# Memory Item"):
            lines[0] = f"# Memory Item {i}"
        renumbered.append("\n".join(lines))
    return MEM_INSTRUCTION + "\n\n" + "\n\n".join(renumbered)


JUDGE_SYSTEM = """You are an expert evaluator of a web-navigation agent on element-selection tasks. You will be given the user's task and the agent's full trajectory (its reasoning and the action it chose at each step). Judge whether the agent's approach, taken as a whole, looks like a successful, coherent completion of the task. You do not have access to the "correct" answer -- judge based on whether the agent's own reasoning and choices are internally consistent, clearly on-topic for the task, and plausibly accomplish what the user asked, the same way you would size up your own past work without an answer key.

Format your response as exactly two lines:
Thoughts: <one or two sentences of reasoning>
Status: "success" or "failure"
"""


def format_trajectory_for_judge(confirmed_task: str, steps: list[dict]) -> str:
    lines = [f"Task: {confirmed_task}", "", "Trajectory:"]
    for step in steps:
        reason = step.get("reason") or "(no reasoning captured)"
        pred = step["prediction"]
        lines.append(
            f"Step {step['step']}: {reason} -> action={pred.get('operation', '')} "
            f"value={pred.get('value', '')!r}"
        )
    return "\n".join(lines)


def judge_success(model_name: str, confirmed_task: str, steps: list[dict]) -> tuple[bool, str]:
    """LLM-as-a-Judge self-assessment (paper Figure 10 / Appendix A.2), adapted to
    Mind2Web's offline trajectory shape: no live final-webpage-state to inspect, only the
    agent's own stated reasoning and chosen actions. Returns (judged_success, raw_response)."""
    prompt = format_trajectory_for_judge(confirmed_task, steps)
    llm = CLIENT_DICT[model_name](model_name=model_name)
    text, _ = llm.one_step_chat(prompt, system_msg=JUDGE_SYSTEM, temperature=0.0)
    # the underlying client can return None text (e.g. an empty/safety-filtered response)
    # instead of raising -- treat that as an unparseable judgement rather than crashing
    text = text or ""
    match = re.search(r"status\s*:\s*[\"']?(success|failure)", text, re.IGNORECASE)
    judged = bool(match) and match.group(1).lower() == "success"
    return judged, text


def format_trajectory_for_induction(confirmed_task: str, steps: list[dict]) -> str:
    lines = [f"**Query:** {confirmed_task}", "", "**Trajectory:**"]
    for step in steps:
        reason = step.get("reason") or "(no reasoning captured)"
        pred = step["prediction"]
        outcome = "correct" if step.get("step_success") else "incorrect"
        lines.append(
            f"<think>\n{reason}\n</think>\n<action>\n"
            f"{pred.get('operation', '')} target={pred.get('backend_node_id')} "
            f"value={pred.get('value', '')!r} ({outcome})\n</action>"
        )
    return "\n\n".join(lines)


def induce_memory(model_name: str, confirmed_task: str, steps: list[dict], judged_success: bool) -> list[str]:
    """Return a list of raw '# Memory Item i ...' text blocks distilled from this task's
    trajectory, using SUCCESSFUL_SI if judge_success() called it a success and FAILED_SI
    otherwise. `judged_success` should come from judge_success(), not ground truth."""
    trajectory = format_trajectory_for_induction(confirmed_task, steps)
    system_msg = SUCCESSFUL_SI if judged_success else FAILED_SI
    llm = CLIENT_DICT[model_name](model_name=model_name)
    text, _ = llm.one_step_chat(trajectory, system_msg=system_msg, temperature=1.0)
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return [block.strip() for block in text.split("\n\n") if block.strip()]


def _strip_fence(text: str) -> str:
    """Strip a whole-response code fence (same handling the reasoningbank/awm/ace branches
    above apply inline), so the block regexes in reme/extract.py and memp/procedure.py see
    clean text."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text


def induce_memory_for_mode(mode: str, model_name: str, confirmed_task: str, steps: list[dict],
                            judged_success: bool, existing: str = "", website: str = "") -> list[str]:
    """Generalizes induce_memory() to the synapse/awm baselines too, mirroring WebArena's
    induce_memory.py --memory_mode branches exactly (same gating: synapse/awm only produce
    something on success; reasoningbank runs on both). `judged_success` -- not ground truth --
    gates these the same way it already gates SUCCESSFUL_SI/FAILED_SI above, for the reason in
    this module's docstring.

    `existing` and `website` are only read by the cer branch, whose prompt is shown the buffer's
    current contents so it can skip what it already distilled."""
    if mode == "reasoningbank":
        return induce_memory(model_name, confirmed_task, steps, judged_success)

    if mode == "synapse":
        # No distillation call -- the raw trajectory IS the "memory".
        if not judged_success:
            return []
        return [format_trajectory_for_induction(confirmed_task, steps)]

    if mode == "awm":
        if not judged_success:
            return []
        trajectory = format_trajectory_for_induction(confirmed_task, steps)
        llm = CLIENT_DICT[model_name](model_name=model_name)
        text, _ = llm.one_step_chat(trajectory, system_msg=AWM_INSTRUCTION + AWM_EXAMPLE, temperature=0.7)
        text = (text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        return [block.strip() for block in text.split("\n\n") if block.strip()]

    if mode == "ace":
        # Reflector step only -- stateless, mirrors WebArena/induce_memory.py's ace branch
        # exactly. Curation (merge into the playbook) happens in run.py via ace/curate.py,
        # same separation of concerns as WebArena's pipeline_memory.py.
        trajectory = format_trajectory_for_induction(confirmed_task, steps)
        system_msg = ACE_REFLECTOR_SI if judged_success else ACE_REFLECTOR_FI
        llm = CLIENT_DICT[model_name](model_name=model_name)
        text, _ = llm.one_step_chat(trajectory, system_msg=system_msg, temperature=0.7)
        text = (text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        return [block.strip() for block in text.split("\n\n") if block.strip()]

    if mode == "reme":
        # ReMe's per-trajectory extraction (paper 3.2): success-pattern recognition on a
        # success, failure analysis on a failure. The comparative analysis needs a success/fail
        # pair and runs separately in run.py via reme/compare.py, for the same reason ACE's
        # curation step lives outside this stateless call.
        trajectory = format_trajectory_for_induction(confirmed_task, steps)
        system_msg = REME_SUCCESS_SI if judged_success else REME_FAILURE_SI
        llm = CLIENT_DICT[model_name](model_name=model_name)
        text, _ = llm.one_step_chat(trajectory, system_msg=system_msg, temperature=0.7)
        return [_strip_fence(text)]

    if mode == "memp":
        # Memp's BUILD step (paper 4.2): one generalized procedure per trajectory. Whether it
        # is appended, skipped, or used to revise an existing procedure in place is the UPDATE
        # step (memp/update.py), applied in run.py -- the paper treats build and update as
        # separate axes, so they stay separate here.
        trajectory = format_trajectory_for_induction(confirmed_task, steps)
        llm = CLIENT_DICT[model_name](model_name=model_name)
        text, _ = llm.one_step_chat(trajectory, system_msg=MEMP_BUILD_SI, temperature=0.7)
        return [_strip_fence(text)]

    if mode == "cer":
        # CER's skills distillation module (paper 3.1). Runs on successes *and* failures: the
        # paper's main setting distills from both, and filtering to successes is its separate
        # CER_success ablation (5.6). `existing` is the buffer's current skills, which the
        # prompt needs so the model can answer "Summarized before" instead of re-distilling
        # (CER's own dedup). Parsing and merging happen in run.py via cer/distill.py, which
        # owns the buffer -- same split as ace/reme/memp above.
        # No dynamics module here: Mind2Web's offline steps carry no URL, so this arm is the
        # paper's "CER - dynamics" variant (5.7). See memory_instruction.py's CER section.
        trajectory = format_trajectory_for_induction(confirmed_task, steps)
        prompt = (
            f"Overall goal: {confirmed_task}\n"
            f"Current website: {website or 'unknown'}\n"
            f"Existing skills:\n{existing or '(none yet)'}\n"
            f"Human user trajectory:\n{trajectory}"
        )
        llm = CLIENT_DICT[model_name](model_name=model_name)
        text, _ = llm.one_step_chat(prompt, system_msg=CER_SKILLS_DISTILL_SI, temperature=0.1)
        return [_strip_fence(text)]

    raise ValueError(f"induce_memory_for_mode: unknown mode {mode!r}")


def append_memory_bank(memory_bank_path: Path, task_id: str, confirmed_task: str, judged_success: bool, memory_items: list[str]) -> None:
    memory_bank_path.parent.mkdir(parents=True, exist_ok=True)
    with memory_bank_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "task_id": task_id,
            "query": confirmed_task,
            "status": "success" if judged_success else "fail",
            "memory_items": memory_items,
        }, ensure_ascii=False) + "\n")
