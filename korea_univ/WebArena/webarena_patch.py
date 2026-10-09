# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runtime compatibility patches for WebArena evaluation.

Two patches, both applied to ``helper_functions`` *and* ``evaluators`` because the
latter imports these functions by value during module initialization:

1. Empty-answer short-circuit. WebArena evaluates string-match tasks after every
   browser action. Some tasks have many fuzzy-match reference fragments, so an
   intermediate click/hover with no submitted answer can otherwise trigger one LLM
   request per fragment. An empty answer cannot match a non-empty reference, so
   decide it locally.

2. Backbone swap: ``llm_fuzzy_match`` / ``llm_ua_match`` go through the Claude Code
   CLI instead of OpenAI. Upstream hardcodes ``gpt-4-1106-preview`` via
   ``generate_from_openai_chat_completion``, which raises if ``OPENAI_API_KEY`` is
   unset -- inside ``env.step()``, so it kills the episode rather than just losing a
   score. Everything else in this repo already runs on the subscription-based CLI
   (``--model ccli-sonnet``), so the ground-truth grader does too and no OpenAI
   account is needed. The prompts below are copied verbatim from upstream
   ``helper_functions`` so the grading criteria are unchanged; only the backbone and
   the unparseable-verdict handling differ (upstream ``assert``s the keyword is
   present, which would again crash the episode -- here an unparseable verdict is a
   miss and is reported once on stderr).

Usage:
    import webarena_patch  # noqa: F401
"""

import sys
from functools import wraps

from webarena.evaluation_harness import evaluators, helper_functions

from utils.clients import CLIENT_DICT

_GRADER_MODEL = "ccli-sonnet"
_SYSTEM_MSG = "You are a helpful assistant"

_client = None
_warned = set()


def _grade(message: str) -> str:
    global _client
    if _client is None:
        _client = CLIENT_DICT[_GRADER_MODEL](model_name=_GRADER_MODEL)
    answer, _ = _client.one_step_chat(message, system_msg=_SYSTEM_MSG, temperature=0.0)
    return (answer or "").lower()


def _unparseable(kind: str, response: str) -> float:
    if kind not in _warned:
        _warned.add(kind)
        print(
            f"[webarena_patch] {kind}: grader reply had no verdict keyword, scoring 0.0 "
            f"(first 200 chars: {response[:200]!r})",
            file=sys.stderr,
            flush=True,
        )
    return 0.0


@wraps(helper_functions.llm_fuzzy_match)
def _llm_fuzzy_match(pred: str, reference: str, question: str) -> float:
    if not pred or not pred.strip():
        return 0.0
    message = (
        "Help a teacher to grade the answer of a student given a question. Keep in mind that the "
        "student may use different phrasing or wording to answer the question. The goal is to "
        "evaluate whether the answer is semantically equivalent to the reference answer.\n"
        f"question: {question}\n"
        f"reference answer: {reference}\n"
        "all the string 'N/A' that you see is a special sequence that means 'not achievable'\n"
        f"student answer: {pred}\n"
        "Conclude the judgement by correct/incorrect/partially correct."
    )
    response = _grade(message)
    if "partially correct" in response or "incorrect" in response:
        return 0.0
    if "correct" in response:
        return 1.0
    return _unparseable("llm_fuzzy_match", response)


@wraps(helper_functions.llm_ua_match)
def _llm_ua_match(pred: str, reference: str, question: str) -> float:
    if not pred or not pred.strip():
        return 0.0
    message = (
        f"task: {question}\n"
        f"actual unachievable reason: {reference}\n"
        f"reported unachievable reason: {pred}\n"
        "The task described above is inherently unachievable due to the reason specified under "
        "'actual unachievable reason'. An individual previously attempted this task and was unable "
        "to complete it. They provided a reason for their failure, which is listed under 'reported "
        "unachievable reason'. Your role is to review both the actual and reported reasons. "
        "Determine if the reported reason aligns with the actual reason, even if implicitly. "
        "If the stated reason is in line with the actual reason, respond with 'same'. Otherwise, "
        "respond with 'different'."
    )
    response = _grade(message)
    if "different" in response:
        return 0.0
    if "same" in response:
        return 1.0
    return _unparseable("llm_ua_match", response)


helper_functions.llm_fuzzy_match = _llm_fuzzy_match
evaluators.llm_fuzzy_match = _llm_fuzzy_match
helper_functions.llm_ua_match = _llm_ua_match
evaluators.llm_ua_match = _llm_ua_match
