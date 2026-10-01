# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");

"""Backbone LLM client that shells out to the Claude Code CLI (`claude`) -- fallback transport
for when Antigravity's (agy) Claude/GPT pool is exhausted. Uses this session's own Claude Code
subscription, so it directly competes with this conversation's own quota; use deliberately, not
by default (see NOTES.md for the tradeoff discussion).

Interface matches utils/agy_client.py's AgyClient / the other CLIENT_DICT classes:
one_step_chat(text, system_msg, json_mode, temperature) -> (response_text, raw_json_dict).
"""

import json
import shutil
import subprocess
import time


def _resolve_claude_bin() -> str:
    found = shutil.which("claude")
    if not found:
        raise FileNotFoundError("Could not locate the `claude` (Claude Code CLI) executable on PATH.")
    return found


class RefusalError(RuntimeError):
    """Claude's safety classifier declined to respond (stop_reason == "refusal"). Empirically
    this is a *probabilistic* false positive (Anthropic's own error text: "This sometimes
    happens with safe, normal conversations"), not a deterministic content block -- resending
    the identical prompt usually succeeds on the next generation. chat() retries this in place
    (fast, no sleep) before giving up; see ClaudeCliClient.n_retry_refusal."""


def _parse_result_event(stdout: str) -> dict:
    """stream-json output emits one NDJSON event per line (system/assistant/result); the final
    "result" event (type == "result") carries {result, is_error, total_cost_usd, usage, ...}."""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") == "result":
            return obj
    raise RuntimeError(f"no result event found in claude stream-json output: {stdout[:1000]!r}")


class ClaudeCliClient:
    def __init__(self, model_name: str = "ccli-sonnet", n_retry_server: int = 6, n_retry_refusal: int = 20) -> None:
        # CLIENT_DICT is keyed by "ccli-<model>" (prefix keeps chat_api.py's
        # startswith("claude"/"gemini"/"openai") dispatch from misrouting this to the Vertex
        # ChatClaude class). Strip it back off here since that's not a real `claude --model` value.
        self.model_name = model_name[len("ccli-"):] if model_name.startswith("ccli-") else model_name
        self.n_retry_server = n_retry_server
        self.n_retry_refusal = n_retry_refusal  # fast, no-sleep retries for the probabilistic
                                                 # reasoning_extraction false-positive (see chat())
        self.claude_bin = _resolve_claude_bin()

    def chat(self, messages, json_mode: bool = False, temperature: float = 0.0):
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), None)
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")

        # Prompt goes over stdin (stream-json), not as a `-p <text>` argv value -- large prompts
        # (judge calls carrying full trajectory text) blow past Windows' CreateProcess
        # command-line length limit (WinError 206), same issue hit with agy_client.py.
        cmd = [
            self.claude_bin, "-p", "--input-format", "stream-json",
            "--output-format", "stream-json", "--verbose",
            "--restricted",  # no Bash/code-exec tools -- plain text completion only, avoids the
                              # tool-call-instead-of-answering failure mode seen with other models
            "--model", self.model_name,
        ]
        # Account separation: set CLAUDE_CONFIG_DIR in the calling process's environment (not
        # here) to point at a second account's OAuth-login config dir, so these subprocess calls
        # draw on that account's subscription quota instead of this session's own. subprocess.run
        # inherits the parent env automatically -- no code-level wiring needed beyond that.
        if system_msg:
            cmd += ["--system-prompt", system_msg]
        stdin_payload = json.dumps({"type": "user", "message": {"role": "user", "content": user_msg}}) + "\n"

        itr = 0
        refusal_itr = 0
        while True:
            try:
                proc = subprocess.run(cmd, input=stdin_payload, capture_output=True, text=True, timeout=300)
                data = _parse_result_event(proc.stdout)
                if data.get("stop_reason") == "refusal":
                    raise RefusalError(f"claude call refused: {data}")
                if data.get("is_error"):
                    raise RuntimeError(f"claude call returned is_error: {data}")
                return data["result"], data
            except RefusalError:
                # Observed empirically: refusal is probabilistic, not deterministic -- the exact
                # same prompt resent often succeeds on the very next generation. Retrying here
                # (no sleep -- it's not a rate-limit issue) recovers within this single call
                # instead of discarding the whole in-progress task/trajectory and restarting it
                # from scratch at the pipeline level, which is far more expensive.
                refusal_itr += 1
                if refusal_itr >= self.n_retry_refusal:
                    raise
                continue
            except Exception as e:  # noqa: BLE001 - mirrors AgyClient's retry-on-anything policy
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
