# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");

"""items.json / experiences.json persistence (EFM_TASK.md 3.2), plus embedding-cache helpers
that reuse memory_management.py's embedding machinery (same model, same cache-file convention
as baseline's query-embedding cache) so "same embedding model" (0-section constraint) is
trivially satisfied by construction."""

import json
import os

import torch

from memory_management import (
    embed_query_with_gemini,
    get_detailed_instruct,
    l2_normalize,
    load_cached_embeddings,
)

_RETRIEVAL_TASK = (
    "Given the prior web navigation queries, your task is to analyze a current query's intent "
    "and select relevant prior queries that could help resolve it."
)


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
    """Embeds `text`, appends {id, text, embedding} to cache_path (same convention as baseline's
    query-embedding cache in memory_management.screening()), returns the raw (1, D) vector."""
    vec = embed_query_with_gemini(text, dimensionality=3072)
    append_cache(vec, cache_path, entry_id, text)
    return vec


def append_cache(vec: torch.Tensor, cache_path: str, entry_id: str, text: str) -> None:
    """Appends an ALREADY-COMPUTED embedding to cache_path without a new API call -- use this
    instead of embed_and_cache whenever the vector was already produced upstream (e.g. insert.py
    already embeds new items once for relation-classification; re-embedding the same text again
    just to persist it would double the embedding API calls for no reason)."""
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": entry_id, "text": text, "embedding": vec.squeeze(0).tolist()}) + "\n")


def embed_query_for_scoring(text: str) -> torch.Tensor:
    """Instruction-wrapped query embedding, matching memory_management.screening()'s scoring
    side exactly. EFM_TASK.md doesn't mention this asymmetry, but baseline's actual retrieval
    scores the current query with this instruction wrapper against plain-embedded cached
    documents -- replicating it here is required for the 5.1 equivalence test to isolate real
    algorithmic differences instead of failing on an embedding-scheme mismatch, and it's the
    more principled choice for real retrieval too (same query representation baseline tuned)."""
    instruction_query = get_detailed_instruct(_RETRIEVAL_TASK, text)
    vec = embed_query_with_gemini(instruction_query, dimensionality=3072)
    return l2_normalize(vec, dim=1)


def load_cache(cache_path: str):
    """Returns (ids, texts, normalized_embeddings_tensor) -- thin re-export for callers that
    only need memory_management's loader without also importing it directly."""
    return load_cached_embeddings(cache_path)


def cosine_to_cache(query_vec: torch.Tensor, cache_emb: torch.Tensor) -> torch.Tensor:
    """Raw [-1, 1] cosine (unlike memory_management.screening()'s *100-scaled version, which
    would drown out EFM's ETA*recency term if reused directly)."""
    if cache_emb.numel() == 0:
        return torch.empty(0)
    q = l2_normalize(query_vec, dim=1)
    return (q @ cache_emb.T).squeeze(0)
