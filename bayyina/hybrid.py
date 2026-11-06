"""Reciprocal rank fusion and the hybrid retriever. You write this file."""
from __future__ import annotations

from bayyina.types import Chunk, Embedder, Hit, Reranker


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Ranks start at 1; score = sum of 1 / (k + rank); by score descending, then ID ascending."""
    raise NotImplementedError


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder, *, k1: float = 1.5,
                 b: float = 0.75, reranker: Reranker | None = None):
        raise NotImplementedError

    def search(self, query: str, top_k: int = 5, candidates: int = 50) -> list[Hit]:
        raise NotImplementedError
