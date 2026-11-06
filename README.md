# W4 task · Bayyina: find the clause, cite the span

DSC Summer Internship 2026 · AI & Deep Tech track · Week 4 · expected time 3–4 hours

## Scenario

Bayyina's juniors waste hours hunting for the termination clause in a forty-page Arabic contract.
Yasmine wants a search layer her team can trust: type a question in Arabic or English, get the
three most relevant passages, each with an exact pointer into the original contract. You will
build it from parts you understand: an Arabic normaliser, a chunker that keeps offsets, BM25
written by hand, fusion with an embedding retriever, an optional re-ranker hook, and a citation
checker. All contracts are synthetic.

## Set up

1. Open this starter (`dsc26-ai-w04-starter`), **Use this template → Create a new repository**,
   owner your own account, name exactly **`dsc26-ai-w04`**.
2. Clone, `uv sync`, `uv run pytest` (the public tests fail until you write the code).

Versions: Python 3.12, numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1 (scipy 1.18.1), pytest 9.1.1.
**No search libraries** (no `rank_bm25`, no FAISS) and nothing else: the grader has nothing else installed.

## Layout

```
dsc26-ai-w04/
├── pyproject.toml          # provided, do not change the dependencies
├── uv.lock
├── bayyina/
│   ├── __init__.py
│   ├── types.py            # provided: Chunk, Hit, Embedder, Reranker (do not edit)
│   ├── embedder.py         # provided: HashingEmbedder, offline stand-in (do not edit)
│   ├── normalize.py        # you write
│   ├── chunking.py         # you write
│   ├── bm25.py             # you write
│   ├── hybrid.py           # you write
│   └── cite.py             # you write
├── data/contracts/*.txt    # provided: 8 synthetic contracts, Arabic, English, bilingual
├── data/queries.jsonl      # provided: 12 public queries with relevant doc ids and spans
├── tests/test_public.py
├── EVAL.md                 # you write: recall@3 for BM25, dense and hybrid on the public queries
└── AI_USAGE.md
```

The grader replaces `types.py` and `embedder.py` with DSC's copies before running.

`bayyina/embedder.py` provides `HashingEmbedder(dim=512, n=3)`: character 3-grams of the
normalised text, hashed into 512 buckets with MD5, L2-normalised. It needs `normalize_ar` from
your `normalize.py`. It is deliberately crude (it rewards shared spelling, not meaning) so that it
runs anywhere; in the squad you replace it with a real model behind the same `embed` method.

## What to build

### `normalize.py`

```python
def normalize_ar(text: str) -> str: ...
def tokenize(text: str) -> list[str]: ...
```

`normalize_ar`: remove U+064B–U+0652 and U+0670; remove tatweel U+0640; map أ إ آ ٱ → ا, ى → ي,
ة → ه; map ٠–٩ and ۰–۹ to 0–9; lowercase; collapse runs of whitespace to one space and strip.
Nothing else: in particular ؤ and ئ stay as they are. `tokenize`: `normalize_ar`, then every match
of the regex `\w+` (Unicode). Franco tokens such as `sha7en` stay intact.

### `chunking.py`

```python
def chunk_text(doc_id: str, text: str, max_chars: int = 500, overlap: int = 50) -> list[Chunk]: ...
```

`ValueError` unless `max_chars > 0` and `0 <= overlap < max_chars`. The output must satisfy, for
any text:

1. `text[c.start:c.end] == c.text` for every chunk, and `c.text` is non-empty, does not start or end
   with whitespace, and has length ≤ `max_chars`;
2. chunk IDs are `f"{doc_id}#{i}"` in order, starts strictly increase;
3. every non-whitespace character of `text` lies inside at least one chunk;
4. consecutive chunks overlap by at most `overlap` characters, and no non-whitespace character lies
   between them;
5. when a chunk cannot reach the end of the text, choose its end among cut points in the second
   half of the `max_chars` window, preferring (in order) just after the last paragraph break `\n\n`,
   just after the last sentence end (`.` `!` `?` `؟` `۔` or `\n`), just after the last space; cut
   mid-word at the window's end only if none of those exists there;
6. a chunk may start mid-word only when the previous chunk ended with a mid-word cut;
7. empty or whitespace-only text returns `[]`.

**Precise meaning of rules 5 and 6** (the grader uses exactly this):

- A chunk starting at `s` *can reach the end* when `len(text) - s <= max_chars`; it then runs to
  the end of the text (minus trailing whitespace).
- Otherwise the window is `text[s : s + max_chars]`. A *cut point* is an offset `s + c` with
  `max_chars // 2 <= c <= max_chars`, where the marker lies wholly inside the window (`\n\n` at
  window positions `c-2, c-1`; a sentence end or a whitespace character at `c-1`). "Space" means
  any whitespace character. Take the largest `c` of the most preferred kind that exists;
  otherwise `c = max_chars` (a hard cut). The chunk's text is `text[s : s + c]` with trailing
  whitespace removed.
- A chunk *starts mid-word* when the character just before its start is not whitespace. A chunk
  *ended with a mid-word cut* when the characters on both sides of its end are not whitespace.
  So after a cut at a paragraph break, sentence end or space, the next chunk starts at the
  beginning of a word (for example the first word inside the overlap, or the first word after
  the cut).
- Overlap is a maximum, not a target: zero overlap is valid.

### `bm25.py`

```python
class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75): ...
    def add(self, chunks: list[Chunk]) -> None: ...
    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]: ...
```

Tokenise chunks and queries with `tokenize`. Each distinct query term counts once. IDF is
`ln(1 + (N − df + 0.5) / (df + 0.5))` where N is the number of chunks indexed. The term score is
`idf × tf × (k1 + 1) / (tf + k1 × (1 − b + b × dl / avgdl))`, with `dl` the chunk's token count and
`avgdl` the mean. Return only chunks with a score above 0, sorted by score descending, then chunk
ID ascending; at most `top_k`. An empty query or an empty index returns `[]`. (The tests call
`add` once per index.)

### `hybrid.py`

```python
def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]: ...

class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder, *, k1: float = 1.5,
                 b: float = 0.75, reranker: Reranker | None = None): ...
    def search(self, query: str, top_k: int = 5, candidates: int = 50) -> list[Hit]: ...
```

`rrf`: ranks start at 1; a chunk's score is the sum of `1 / (k + rank)` over the lists it appears
in; sort by score descending, then ID ascending. `search`: blank query → `[]`. Take the top
`candidates` from BM25 and the top `candidates` by dot product with the embedded query (ties by
chunk ID), fuse the two ID lists with `rrf`, keep the first `candidates`, and build `Hit`s whose
offsets and text come from the original chunks and whose score is the RRF score. If a `reranker`
is given, call it once with the query and the fused hits, sort by its scores descending (ties by
chunk ID), and put the reranker's score in `Hit.score`. Return the first `top_k`.

Use the embedder you are given (the grader passes embedders you have never seen). "By chunk ID"
always means plain string order, so `doc#10` comes before `doc#2`.

### `cite.py`

```python
def verify_citation(quote: str, document: str, start: int, end: int) -> bool: ...
def format_citation(hit: Hit, max_quote: int = 160) -> str: ...
```

`verify_citation` is true only when `0 <= start < end <= len(document)`, the quote is not blank,
and the quote (whitespace collapsed) occurs inside `document[start:end]` (whitespace collapsed).
"Whitespace collapsed" means `" ".join(s.split())`. `format_citation` returns
`[doc_id:start-end] "quote"` where the quote is the hit's text with whitespace collapsed; when it is
longer than `max_quote`, take its first `max_quote` characters, cut back to the last space inside
them (dropping the space), and append `…`.

## `EVAL.md` and `AI_USAGE.md`

`EVAL.md`: recall@3 for BM25 only, dense only and hybrid on the 12 public queries (a query counts
as found when one of the top 3 hits overlaps one of its `relevant` ranges in the same document),
with `chunk_text(..., max_chars=500, overlap=50)` and `HashingEmbedder()`. `AI_USAGE.md`: tools and
what for; one place where AI-written chunking or BM25 code broke one of the spec's rules, and how
you found it; your recall@3 table and one sentence on why hybrid did or did not win. Both are read
by your Track Mentor, not graded by tests.

## How grading works

Push, then **Submit for grading** on DSC Hex. The grader clones that commit, restores `types.py`
and `embedder.py`, and runs the hidden tests on Python 3.12 with the versions above, network off,
60 seconds per test, 5 minutes in all. Score 0–100 with feedback per test name; each test is all
or nothing; up to five runs per task per day; the best attempt counts.

| Test name | What it checks | Points |
|---|---|---|
| `normalize_rules` | Each normalisation rule in isolation, including U+0670, ٱ, Persian digits, mixed scripts | 9 |
| `tokenize_mixed_scripts` | Arabic, English, Franco, digits, punctuation | 4 |
| `chunk_offsets_property` | 300 random texts, random `max_chars` and `overlap`: rules 1 and 2 | 10 |
| `chunk_coverage_and_overlap_property` | Same texts: rules 3 and 4 | 8 |
| `chunk_prefers_structure` | Paragraph break over sentence end over space; rules 5 and 6 | 6 |
| `chunk_edge_cases` | Empty, whitespace, one 2,000-character word, bad arguments raise | 5 |
| `bm25_matches_reference_scores` | Scores equal the formula within 1e-9 on a 40-chunk corpus, three (k1, b) pairs | 12 |
| `bm25_uses_normalisation` | أحكام / احكام, diacritics, Arabic-Indic digits all match | 5 |
| `bm25_ordering_and_limits` | Tie-break by ID, `top_k`, repeated query terms, empty query | 5 |
| `rrf_correctness` | Items missing from one list, no duplicates, custom `k`, tie order | 7 |
| `hybrid_uses_dense_path` | A chunk only an (unseen) embedder can find must appear | 8 |
| `hybrid_hits_carry_offsets` | Every hit's text equals the source span; scores are RRF scores | 5 |
| `reranker_hook` | Called exactly once; order and `Hit.score` from the reranker | 5 |
| `retrieval_quality_hidden_queries` | Recall@3 ≥ 0.65 on 20 hidden queries over the provided contracts, using your `chunk_text(..., 500, 50)`, your `HybridRetriever` and `HashingEmbedder` | 7 |
| `citation_checks` | `verify_citation` boundaries and whitespace; `format_citation` truncation | 4 |

Late policy: ×1.0 within 72 hours of the session starting, ×0.6 in the following 7 days, ×0.5
after that until Thu 3 Dec 23:59. During W5 (7–13 Nov) you can resubmit at ×1.0.

If you are stuck for more than an hour, ask in the AI track channel on Discord or ask Hexa
rather than spending grader runs.
