# test_embedding.py
import os, sys
sys.path.insert(0, "WebArena")

print("API key set:", bool(os.environ.get("GOOGLE_API_KEY")))
print("USE_VERTEXAI:", os.environ.get("GOOGLE_GENAI_USE_VERTEXAI"))

from memory_management import embed_query_with_gemini, get_embeddings

vec = embed_query_with_gemini("find a red running shoe under $50")
print("query vec:", vec.shape, vec.dtype)

embs = get_embeddings(["first memory item", "second memory item"])
print("batch count:", len(embs), "dim:", len(embs[0]))
print("OK")