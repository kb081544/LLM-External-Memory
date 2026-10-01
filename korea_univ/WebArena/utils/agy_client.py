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

"""Backbone LLM client that shells out to the Antigravity CLI (`agy`) instead of calling a
metered API directly. Used to route agent/judge/extraction/EFM calls through a subscription-based
transport after repeatedly hitting the Gemini API's free-tier quota walls. `agy` has no native
system-prompt flag, so system_msg is prepended into the single `-p` prompt string -- functionally
equivalent to a system+user completion for a one-shot call.

Interface matches the other CLIENT_DICT classes in this file's siblings
(GEMINI_Client/CLAUDE_Client/GPT_Client in clients.py): one_step_chat(text, system_msg,
json_mode, temperature) -> (response_text, raw_json_dict).
"""

import json
import os
import shutil
import subprocess
import time

# shutil.which("agy") fails here even though `agy` works fine when typed into an interactive
# terminal -- the install dir isn't on PATH for non-interactive/subprocess-launched sessions
# (observed during the 5.3 smoke test: subprocess.run(["agy", ...]) raised WinError 2, file not
# found). Fall back to the known Windows install location before giving up.
_AGY_FALLBACK_PATHS = [
    os.path.expandvars(r"%LOCALAPPDATA%\agy\bin\agy.exe"),
]


def _resolve_agy_bin() -> str:
    found = shutil.which("agy")
    if found:
        return found
    for p in _AGY_FALLBACK_PATHS:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "Could not locate the `agy` (Antigravity CLI) executable on PATH or at the known "
        f"fallback location(s): {_AGY_FALLBACK_PATHS}. Install Antigravity CLI or update "
        "_AGY_FALLBACK_PATHS in utils/agy_client.py."
    )


def _parse_result_event(stdout: str) -> dict:
    """stream-json output mode emits one NDJSON event per line (init/step_update/result); the
    final "result" event carries the same {status, response, usage, ...} shape the old plain
    --output-format json mode returned as the whole stdout."""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("event") == "result":
            return obj["result"]
    raise RuntimeError(f"no result event found in agy stream-json output: {stdout[:1000]!r}")


class AgyClient:
    def __init__(self, model_name: str = "agy-claude-sonnet-4-6", n_retry_server: int = 6) -> None:
        # CLIENT_DICT is keyed by "agy-<model>" (the "agy-" prefix keeps chat_api.py's
        # startswith("claude"/"gemini"/"openai") dispatch from misrouting to the Vertex clients).
        # Strip it back off here since that's not a real `agy --model` value.
        self.model_name = model_name[len("agy-"):] if model_name.startswith("agy-") else model_name
        self.n_retry_server = n_retry_server
        self.agy_bin = _resolve_agy_bin()  # resolved lazily (per-instance), not at import time

    def chat(self, messages, json_mode: bool = False, temperature: float = 0.0):
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        combined = f"{system_msg}\n\n{user_msg}" if system_msg else user_msg

        # Large prompts (judge calls carrying full trajectory/axtree text) blow past Windows'
        # CreateProcess command-line length limit when passed as a `-p` argv value (WinError 206,
        # hit during the 5.3 smoke test). --input-format stream-json reads the prompt from stdin
        # instead, which has no such limit. --print needs an explicit (empty) value or it greedily
        # consumes the next flag as its prompt.
        cmd = [
            self.agy_bin, "--print=", "--input-format", "stream-json",
            "--output-format", "stream-json", "--model", self.model_name,
        ]
        stdin_payload = json.dumps({"event": "user", "message": {"role": "user", "content": combined}}) + "\n"

        itr = 0
        while True:
            try:
                proc = subprocess.run(cmd, input=stdin_payload, capture_output=True, text=True, timeout=300)
                data = _parse_result_event(proc.stdout)
                if data.get("status") != "SUCCESS":
                    raise RuntimeError(f"agy call did not succeed: {data}")
                return data["response"], data
            except Exception as e:  # noqa: BLE001 - mirrors the broad except+backoff already used
                # by ChatGemini._call in agents/legacy/utils/chat_api.py for the same reason:
                # transient CLI/subprocess/parse failures should retry, not crash the pipeline.
                if itr == self.n_retry_server - 1:
                    raise
                time.sleep(min(2 ** itr, 60))
                itr += 1

    def one_step_chat(
        self, text, system_msg: str = None, json_mode: bool = False, temperature: float = 0.0
    ):
        messages = []
        if system_msg is not None:
            messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": text})
        return self.chat(messages, json_mode=json_mode, temperature=temperature)
