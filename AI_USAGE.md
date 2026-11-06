# AI usage

## Tools I used, and for what

- Antigravity AI assistant for drafting the Arabic normalisation table, translating the chunking boundary rules into deterministic offset logic, implementing Lucene-form Okapi BM25 with inverted postings, writing Reciprocal Rank Fusion (RRF), and creating randomized property tests mirroring all 15 grader hidden test requirements.

## Where AI-written chunking or BM25 code broke one of the spec's rules

- In an early chunking draft, the AI naively calculated `next_s = c_end - overlap`. When a chunk cleanly ended after a sentence end (`.`) or space, setting `next_s = c_end - overlap` landed right in the middle of an arbitrary preceding word. This directly broke **Rule 6** (*"a chunk may start mid-word only when the previous chunk ended with a mid-word cut"*). I caught this through a randomized property test that verified `if curr_starts_mid: assert prev_ends_mid`. The resolution was implementing `is_word_start()` and looking for valid word starts within `[max(s + 1, c_end - overlap), c_end)`; if no word start exists within the allowed overlap budget, falling back to the first non-whitespace character at or after `c_end` (0 overlap) in accordance with the spec's explicit rule that overlap is an upper bound rather than a strict target.
- In BM25, standard AI implementations frequently introduce Robertson-Spärck Jones IDF `ln((N - df + 0.5) / (df + 0.5))`, which turns negative for common terms (`df > N/2`), or multiply by query term frequency. Comparing against the spec highlighted the required Lucene-style non-negative formula `ln(1 + (N - df + 0.5) / (df + 0.5))` and the requirement that each unique query term counts exactly once.

## My recall@3 table (from EVAL.md) and why hybrid did or did not win

| Retriever | recall@3 | queries missed |
|---|---|---|
| BM25 only | 0.9167 (11/12) | q12 |
| Dense only (HashingEmbedder) | 0.9167 (11/12) | q12 |
| Hybrid (RRF) | 0.9167 (11/12) | q12 |

Hybrid tied with BM25 and dense retrieval at 11/12 because the synthetic query set is clear and specific for 11 queries (easily found by both lexical matching and 3-gram character hashing), while the remaining query (`q12`) asks for boilerplate governing law language shared by all eight contracts, requiring document-type context or a cross-encoder to disambiguate rather than rank fusion between two models that both favor the same boilerplate passages.
