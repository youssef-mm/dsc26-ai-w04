# Retrieval evaluation

Setup: all 8 contracts in `data/contracts/`, `chunk_text(..., max_chars=500, overlap=50)`,
`HashingEmbedder()`, the 12 queries in `data/queries.jsonl`. A hit is relevant when its span
overlaps one of the query's `relevant` ranges in the same document.

| Retriever | recall@3 | queries missed |
|---|---|---|
| BM25 only | 0.9167 (11/12) | q12 |
| Dense only (HashingEmbedder) | 0.9167 (11/12) | q12 |
| Hybrid (RRF) | 0.9167 (11/12) | q12 |

### Analysis and Next Steps

The single miss across all three retrievers was `q12` ("governing law and competent court for the agency agreement"). Every commercial contract in the corpus contains a boilerplate governing law and jurisdiction clause; because `lease_en_02` and `nda_en_06` contain dense phrasing matching "governing law", their chunks crowd the top 3 ranks, pushing the target clause in `agency_bi_08` down to rank 9.

To improve retrieval and resolve document-specific boilerplate queries:
1. **Contextual retrieval chunk headers (Anthropic)**: Prepend document-level context (e.g., `Document: agency_bi_08 (Commercial Agency Agreement)`) to each chunk before indexing, enabling BM25 and vector search to match "agency agreement" directly to the right contract.
2. **Semantic bi-encoder**: Replace the character 3-gram hashing stand-in with a true multilingual dense model (`intfloat/multilingual-e5-small` or `BAAI/bge-m3`) using query/passage prefixes.
3. **Cross-encoder re-ranking**: Deploy a cross-encoder (`BAAI/bge-reranker-v2-m3`) over the top 20 candidates retrieved by RRF, allowing joint cross-attention between query terms and candidate spans.
