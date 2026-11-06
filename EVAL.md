# Retrieval evaluation

<!-- Read by your Track Mentor, not graded by tests. Replace the comments. -->

Setup: all 8 contracts in `data/contracts/`, `chunk_text(..., max_chars=500, overlap=50)`,
`HashingEmbedder()`, the 12 queries in `data/queries.jsonl`. A hit is relevant when its span
overlaps one of the query's `relevant` ranges in the same document.

| Retriever | recall@3 | queries missed |
|---|---|---|
| BM25 only | | |
| Dense only (HashingEmbedder) | | |
| Hybrid (RRF) | | |

<!-- Two or three sentences: what the misses have in common, and what you would try next
     (a real embedding model, contextual chunk headers, a re-ranker, different chunk size). -->
