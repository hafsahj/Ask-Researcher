"""A minimal FAISS-backed vector store for retrieval."""
from dataclasses import dataclass

import faiss
import numpy as np

from .ingest import Chunk


@dataclass
class SearchResult:
    chunk: Chunk
    score: float  # cosine similarity, higher is more relevant


class VectorStore:
    def __init__(self, dim: int):
        # Inner product on L2-normalized vectors == cosine similarity.
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks and embeddings must have the same length")
        normalized = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        self.index.add(normalized)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[SearchResult]:
        if self.index.ntotal == 0:
            return []
        query = query_embedding.reshape(1, -1)
        query = query / (np.linalg.norm(query) + 1e-8)
        scores, indices = self.index.search(query, min(k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(SearchResult(chunk=self.chunks[idx], score=float(score)))
        return results

    def __len__(self) -> int:
        return self.index.ntotal
