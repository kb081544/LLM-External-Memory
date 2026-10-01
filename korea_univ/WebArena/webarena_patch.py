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

WebArena evaluates string-match tasks after every browser action. Some tasks
have many fuzzy-match reference fragments, so an intermediate click/hover with
no submitted answer can otherwise trigger one LLM request per fragment. An
empty answer cannot match a non-empty reference, so short-circuit it locally.

This patch is applied to both ``helper_functions`` and ``evaluators`` because
the latter imports ``llm_fuzzy_match`` by value during module initialization.

Usage:
    import webarena_patch  # noqa: F401
"""

from functools import wraps

from webarena.evaluation_harness import evaluators, helper_functions


_original_llm_fuzzy_match = helper_functions.llm_fuzzy_match


@wraps(_original_llm_fuzzy_match)
def _skip_empty_llm_fuzzy_match(pred: str, reference: str, question: str) -> float:
    """Return a definite miss without calling an LLM for an empty answer."""
    if not pred or not pred.strip():
        return 0.0
    return _original_llm_fuzzy_match(pred, reference, question)


helper_functions.llm_fuzzy_match = _skip_empty_llm_fuzzy_match
evaluators.llm_fuzzy_match = _skip_empty_llm_fuzzy_match
