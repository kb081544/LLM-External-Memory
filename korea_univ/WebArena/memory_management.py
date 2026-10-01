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


'''
Code Adapted from https://github.com/YerbaPage/SWE-Exp/blob/main/moatless/experience/exp_agent/select_agent.py
'''

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import logging
logger = logging.getLogger(__name__)
from google import genai
from google.genai.types import EmbedContentConfig
#import vertexai
#from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
# TextEmbeddingModel requires a regional endpoint; "global" doesn't work.
#vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"),location="us-central1")
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


def get_embeddings(texts: list) -> list:
    """
    Get embeddings for a list of texts using Google GenAI.
    """
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
        config=EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=3072,
            title="Memory Embeddings",
        ),
    )
    return [item.values for item in response.embeddings]

def l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(x, p=2, dim=dim)

def embed_query_with_gemini(query: str, dimensionality: int = 3072) -> torch.Tensor:
    """Returns (1, D) torch tensor (on CPU)."""
    resp = client.models.embed_content(
        model="gemini-embedding-001",
        contents=[query],
        config=EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=dimensionality,
        ),
    )
    return torch.tensor([resp.embeddings[0].values], dtype=torch.float32)


def load_cached_embeddings(path: str) -> Tuple[List[str], List[str], torch.Tensor]:
    """
    Load cached embeddings from JSONL.
    Returns: ids, texts, torch.Tensor (N, D) normalized
    Each line must contain keys: id, text, embedding
    """
    ids, texts, vecs = [], [], []
    if not os.path.exists(path):
        logger.warning(f"Cache file not found: {path}, creating an empty cache.")
        open(path, "w").close()  # create an empty file
        return ids, texts, torch.empty(0)

    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            ids.append(obj["id"])
            texts.append(obj.get("text", ""))
            vecs.append(obj["embedding"])

    if len(vecs) == 0:
        return ids, texts, torch.empty(0)

    emb = torch.tensor(vecs, dtype=torch.float32)  # (N, D)
    emb = l2_normalize(emb, dim=1)

    return ids, texts, emb

def average_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def formalize(queries):
    tmp = []
    ids = []
    for id, data in enumerate(queries):
        ids.append(id)
        tmp.append(data)
    return tmp, ids

def select_memory(n: int,
                  reasoning_bank: List[Dict],
                  cur_query: str,
                  task_id: str = None,
                  cache_path: str = "./memories/embeddings.jsonl",
                  prefer_model: str = "gemini") -> Dict:
    """
    Returns a dict of top-n items by ID -> (optionally) original metadata.
    This uses ONLY the cached embeddings; it does not recompute them.
    """
    if n > 10:
        logger.error("the number of return experiences shouldn't be greater than 10")

    id2score, ordered_ids = screening(cur_query=cur_query,
                                      task_id=task_id,
                                      cache_path=cache_path,
                                      prefer_model=prefer_model)

    if not ordered_ids:
        return {}

    top_ids = ordered_ids[:n]

    # optional: map back to your in-memory store if you have it
    # below assumes your cache ids correspond 1:1 to indices in reasoning_bank
    out = []
    for sid in top_ids:
        # find the corresponding reasoning bank entry, with reasoning_bank["task_id"] == sid
        for i, item in enumerate(reasoning_bank):
            if item["task_id"] == sid:
                out.append(reasoning_bank[i])
                break
    return out

def screening(cur_query: str,
              cache_path: str,
              task_id: str = None,
              prefer_model: str = "",) -> Tuple[List[Tuple[str, float]], List[str]]:
    """
    Compute similarity of current query against cached embeddings, optionally append the query embedding to cache.
    """
    cache_ids, cache_texts, cache_emb = load_cached_embeddings(cache_path)

    # choose embedding method to match the cache
    use_qwen = "Qwen" in prefer_model

    if use_qwen:
        q_vec = embed_query_with_qwen(cur_query)
    else:
        q_vec = embed_query_with_gemini(cur_query, dimensionality=3072)

    # write current query embeddings to cache
    record = {
        "id": task_id,
        "text": cur_query,
        "embedding": q_vec.squeeze(0).tolist(),
    }
    with open(cache_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    logger.info(f"Appended new query embedding to cache: webarena.{task_id}")

    if len(cache_emb) == 0:
        logger.warning(f"No cached embeddings found in {cache_path}.")
        return [], []

    # add instruction-aware embedding for calculation
    task = "Given the prior web navigation queries, your task is to analyze a current query's intent and select relevant prior queries that could help resolve it."

    instruction_query = get_detailed_instruct(task, cur_query)
    instruct_vec = embed_query_with_gemini(instruction_query, dimensionality=3072)
    instruct_vec = l2_normalize(instruct_vec, dim=1)

    # Calculate similarity scores for embeddings and current query
    scores = (instruct_vec @ cache_emb.T).squeeze(0) * 100.0  # (N,)
    id2score = list(zip(cache_ids, scores.tolist()))
    id2score.sort(key=lambda x: x[1], reverse=True)

    return id2score, [str(i) for i, _ in id2score]


def log_retrieval_event(usage_log_path: str, query_task_id: str, retrieved_task_ids: List[str]) -> None:
    """Append-only record of which memory bank entries were retrieved for a task, so a
    later pass (see memory_prune.py) can join this against the task's eventual outcome and
    compute a per-entry utility score. Only called for modes that opt into usage tracking
    (reasoningbank_pruned) -- every other mode never passes a usage_log_path, so this never
    runs for them."""
    os.makedirs(os.path.dirname(usage_log_path), exist_ok=True)
    with open(usage_log_path, "a") as f:
        f.write(json.dumps({
            "query_task_id": query_task_id,
            "retrieved_task_ids": retrieved_task_ids,
        }) + "\n")