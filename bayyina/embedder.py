"""Offline stand-in for an embedding model. Provided by DSC. Do not edit.

Character 3-gram hashing over normalised text. It is crude, but it runs on CPU
with no download, and it rewards spelling overlap, which is enough to show why
fusing it with BM25 helps. In your squad you would swap it for a real model
such as intfloat/multilingual-e5-small.
"""
from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from bayyina.normalize import normalize_ar


class HashingEmbedder:
    def __init__(self, dim: int = 512, n: int = 3):
        self.dim, self.n = dim, n

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim)
        t = f" {normalize_ar(text)} "
        for i in range(len(t) - self.n + 1):
            h = hashlib.md5(t[i:i + self.n].encode("utf-8")).digest()
            v[int.from_bytes(h[:4], "little") % self.dim] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim))
