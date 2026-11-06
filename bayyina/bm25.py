"""BM25, written by hand. You write this file (no search libraries)."""
from __future__ import annotations

from bayyina.types import Chunk


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        raise NotImplementedError

    def add(self, chunks: list[Chunk]) -> None:
        raise NotImplementedError

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """[(chunk_id, score), ...] with score > 0, by score descending then chunk ID ascending."""
        raise NotImplementedError
