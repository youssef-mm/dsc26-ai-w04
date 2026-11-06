"""Reciprocal rank fusion and the hybrid retriever."""
from __future__ import annotations

import numpy as np

from bayyina.bm25 import BM25Index
from bayyina.types import Chunk, Embedder, Hit, Reranker


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Ranks start at 1; score = sum of 1 / (k + rank); by score descending, then ID ascending."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        seen = set()
        for rank, cid in enumerate(ranking, start=1):
            if cid in seen:
                continue
            seen.add(cid)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        embedder: Embedder,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        reranker: Reranker | None = None,
    ):
        self.chunks = list(chunks)
        self.chunk_by_id = {c.chunk_id: c for c in chunks}
        self.embedder = embedder
        self.reranker = reranker

        self.bm25 = BM25Index(k1=k1, b=b)
        self.bm25.add(chunks)

        if self.chunks:
            self.chunk_embeddings = self.embedder.embed([c.text for c in self.chunks])
        else:
            self.chunk_embeddings = np.zeros((0, 0))

    def search(self, query: str, top_k: int = 5, candidates: int = 50) -> list[Hit]:
        """Search query using hybrid retrieval (BM25 + Dense) fused with RRF."""
        if not query or not query.strip() or not self.chunks:
            return []

        # 1. Top candidates from BM25
        bm25_results = self.bm25.search(query, top_k=candidates)
        bm25_ids = [cid for cid, _ in bm25_results]

        # 2. Top candidates by dot product with the embedded query (ties by chunk ID)
        q_emb = self.embedder.embed([query])[0]
        dot_products = np.dot(self.chunk_embeddings, q_emb)
        dense_scored = [
            (float(dot_products[i]), self.chunks[i].chunk_id)
            for i in range(len(self.chunks))
        ]
        dense_scored.sort(key=lambda x: (-x[0], x[1]))
        dense_ids = [cid for _, cid in dense_scored[:candidates]]

        # 3. Fuse with RRF, keep first candidates
        fused = rrf([bm25_ids, dense_ids], k=60)
        fused_candidates = fused[:candidates]

        # 4. Build Hits with RRF scores
        hits = [
            Hit(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                start=c.start,
                end=c.end,
                text=c.text,
                score=rrf_score,
            )
            for cid, rrf_score in fused_candidates
            for c in [self.chunk_by_id[cid]]
        ]

        # 5. Optional reranker hook
        if self.reranker is not None and hits:
            rerank_scores = self.reranker(query, hits)
            reranked_hits = [
                Hit(
                    chunk_id=h.chunk_id,
                    doc_id=h.doc_id,
                    start=h.start,
                    end=h.end,
                    text=h.text,
                    score=float(rerank_scores[i]),
                )
                for i, h in enumerate(hits)
            ]
            reranked_hits.sort(key=lambda h: (-h.score, h.chunk_id))
            hits = reranked_hits

        return hits[:top_k]
