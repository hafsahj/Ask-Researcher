"""Dense embeddings via sentence-transformers.

Runs locally within whatever compute the app is deployed on (a Hugging Face
Space, in this project's case) rather than calling an external API - keeps
embedding free of per-call rate limits, only the generation step needs an
API token. Model is small enough (~80MB) to load comfortably on a free-tier
CPU Space.
"""
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of strings, returns a (len(texts), dim) float32 array."""
    model = get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.astype("float32")
