# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");

"""items.json / experiences.json persistence (EFM_TASK.md 3.2), plus embedding-cache helpers
that reuse memory.py's embedding machinery (same model/cache-file convention as the working
`reasoningbank` arm's query-embedding cache) so "same embedding model" is satisfied by
construction. Unlike WebArena's baseline (memory_management.screening()), Mind2Web's
`memory.select_memory()` scores with a *plain* embedding of the current query -- no
instruction-wrap asymmetry -- so this module doesn't need a separate embed_query_for_scoring();
one plain embed_text() is used for both caching and scoring, matching baseline exactly."""

import json
import os
from pathlib import Path

import torch
from google import genai

import memory as memory_lib

_client = genai.Client()


def embed_text(text: str) -> torch.Tensor:
    """Plain embedding via memory.py's embed_query -- same model/dimensionality baseline uses."""
    return memory_lib.embed_query(_client, text)


def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def next_id(existing: dict, prefix: str) -> str:
    n = len(existing)
    while f"{prefix}_{n:04d}" in existing:
        n += 1
    return f"{prefix}_{n:04d}"


def embed_and_cache(text: str, cache_path: str, entry_id: str) -> torch.Tensor:
    """Embeds `text`, appends {id, text, embedding} to cache_path, returns the raw (1, D) vector."""
    vec = embed_text(text)
    append_cache(vec, cache_path, entry_id, text)
    return vec


def append_cache(vec: torch.Tensor, cache_path: str, entry_id: str, text: str) -> None:
    """Appends an ALREADY-COMPUTED embedding to cache_path without a new API call -- use this
    instead of embed_and_cache whenever the vector was already produced upstream (insert.py
    embeds new items once for relation-classification; re-embedding the same text again just to
    persist it would double the embedding API calls for no reason -- see WebArena NOTES.md)."""
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": entry_id, "text": text, "embedding": vec.squeeze(0).tolist()}) + "\n")


def load_cache(cache_path: str):
    """Returns (ids, texts, normalized_embeddings_tensor). memory.py's own loader only returns
    (ids, emb) -- texts aren't used anywhere in the EFM algorithm, so a dummy empty list is
    returned in that slot to keep the 3-tuple interface the ported insert.py/retrieve.py expect."""
    ids, emb = memory_lib.load_cached_embeddings(Path(cache_path))
    return ids, [""] * len(ids), emb


def cosine_to_cache(query_vec: torch.Tensor, cache_emb: torch.Tensor) -> torch.Tensor:
    """Raw [-1, 1] cosine (not baseline's *100-scaled version, which would drown out EFM's
    ETA*recency term if reused directly)."""
    if cache_emb.numel() == 0:
        return torch.empty(0)
    q = memory_lib.l2_normalize(query_vec, dim=1)
    return (q @ cache_emb.T).squeeze(0)
