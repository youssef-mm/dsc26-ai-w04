"""BM25, written by hand."""
from __future__ import annotations

from collections import Counter
import math

from bayyina.normalize import tokenize
from bayyina.types import Chunk


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self.doc_len: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.df: dict[str, int] = {}
        self.avgdl: float = 0.0
        self.N: int = 0

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self.chunks = list(chunks)
        self.N = len(chunks)
        self.doc_len = []
        self.postings = {}
        self.df = {}
        total_len = 0

        for i, c in enumerate(chunks):
            tokens = tokenize(c.text)
            dl = len(tokens)
            self.doc_len.append(dl)
            total_len += dl
            counts = Counter(tokens)
            for t, tf in counts.items():
                if t not in self.postings:
                    self.postings[t] = []
                self.postings[t].append((i, tf))
                self.df[t] = self.df.get(t, 0) + 1

        self.avgdl = total_len / self.N if self.N > 0 else 0.0

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """[(chunk_id, score), ...] with score > 0, by score descending then chunk ID ascending."""
        if not query or not query.strip() or self.N == 0:
            return []

        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        unique_terms = set(q_tokens)
        doc_scores: dict[int, float] = {}

        for t in unique_terms:
            if t not in self.df:
                continue
            df = self.df[t]
            # IDF formula: ln(1 + (N - df + 0.5) / (df + 0.5))
            idf = math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))
            for i, tf in self.postings[t]:
                dl = self.doc_len[i]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl if self.avgdl > 0 else 1.0))
                term_score = idf * tf * (self.k1 + 1.0) / denom
                doc_scores[i] = doc_scores.get(i, 0.0) + term_score

        results = [(self.chunks[i].chunk_id, s) for i, s in doc_scores.items() if s > 0]
        results.sort(key=lambda x: (-x[1], x[0]))
        return results[:top_k]
