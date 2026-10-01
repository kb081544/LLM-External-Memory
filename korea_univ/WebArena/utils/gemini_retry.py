"""Monkey-patch google.genai Models methods with exponential-backoff retry.

Import this module once per process before any Gemini call. Re-imports are
idempotent. Patches:
    - Models.generate_content
    - Models.generate_content_stream
    - Models.embed_content
    - AsyncModels.generate_content
    - AsyncModels.generate_content_stream
    - AsyncModels.embed_content

Retries on 429 (RESOURCE_EXHAUSTED), 500 (INTERNAL), 503 (UNAVAILABLE),
504 (DEADLINE_EXCEEDED) with exponential backoff + jitter. Other exceptions
propagate immediately.
"""
from __future__ import annotations

import asyncio
import functools
import logging
import random
import time

log = logging.getLogger("gemini_retry")

MAX_RETRIES = 6
BASE_DELAY = 4.0
MAX_DELAY = 60.0

_RETRYABLE_MARKERS = (
    "RESOURCE_EXHAUSTED",
    "grpc_status:8",
    "429",
    "UNAVAILABLE",
    "grpc_status:14",
    "503",
    "DEADLINE_EXCEEDED",
    "grpc_status:4",
    "504",
    "INTERNAL",
    "grpc_status:13",
    "500",
)


def _is_retryable(exc: BaseException) -> bool:
    msg = f"{type(exc).__name__}: {exc}"
    return any(m in msg for m in _RETRYABLE_MARKERS)


def _sleep_for(attempt: int) -> float:
    return min(MAX_DELAY, BASE_DELAY * (2 ** attempt)) + random.uniform(0, 1)


def _wrap_sync(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        attempt = 0
        while True:
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if not _is_retryable(e) or attempt >= MAX_RETRIES:
                    raise
                delay = _sleep_for(attempt)
                log.warning(
                    "Gemini %s failed (%s), retrying in %.1fs [attempt %d/%d]",
                    fn.__name__, type(e).__name__, delay, attempt + 1, MAX_RETRIES,
                )
                time.sleep(delay)
                attempt += 1
    return wrapped


def _wrap_async(fn):
    @functools.wraps(fn)
    async def wrapped(*args, **kwargs):
        attempt = 0
        while True:
            try:
                return await fn(*args, **kwargs)
            except Exception as e:
                if not _is_retryable(e) or attempt >= MAX_RETRIES:
                    raise
                delay = _sleep_for(attempt)
                log.warning(
                    "Gemini %s failed (%s), retrying in %.1fs [attempt %d/%d]",
                    fn.__name__, type(e).__name__, delay, attempt + 1, MAX_RETRIES,
                )
                await asyncio.sleep(delay)
                attempt += 1
    return wrapped


_PATCHED = False


def patch() -> None:
    global _PATCHED
    if _PATCHED:
        return
    from google.genai import models

    for name in ("generate_content", "generate_content_stream", "embed_content"):
        orig = getattr(models.Models, name)
        setattr(models.Models, name, _wrap_sync(orig))
        if hasattr(models, "AsyncModels"):
            orig_async = getattr(models.AsyncModels, name)
            setattr(models.AsyncModels, name, _wrap_async(orig_async))

    _PATCHED = True
    log.info("gemini_retry: patched Models/AsyncModels with exponential backoff")


patch()
