"""Shared types. Provided by DSC. Do not edit."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: str      # f"{doc_id}#{i}", i = 0, 1, 2 ... in document order
    start: int         # offset into the ORIGINAL document text
    end: int           # exclusive; text == document[start:end]
    text: str


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    doc_id: str
    start: int
    end: int
    text: str
    score: float


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dim) with L2-normalised rows."""


Reranker = Callable[[str, list[Hit]], list[float]]   # higher score = more relevant
