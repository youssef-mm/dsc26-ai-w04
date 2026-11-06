import math
import random
import re
from typing import Sequence
import numpy as np
import pytest

from bayyina.bm25 import BM25Index
from bayyina.chunking import chunk_text
from bayyina.cite import format_citation, verify_citation
from bayyina.embedder import HashingEmbedder
from bayyina.hybrid import HybridRetriever, rrf
from bayyina.normalize import normalize_ar, tokenize
from bayyina.types import Chunk, Hit


# 1. normalize_rules (9 pts)
def test_normalize_rules_isolation():
    # U+0670 superscript alef
    assert normalize_ar("هٰذَا") == "هذا"
    assert normalize_ar("الرَّحْمٰن") == "الرحمن"

    # ٱ (wasla) -> ا
    assert normalize_ar("ٱلله") == "الله"
    assert normalize_ar("وَٱسْتَغْفِرْ") == "واستغفر"

    # Persian / Eastern Arabic-Indic digits ۰-۹ -> 0-9
    assert normalize_ar("۰۱۲۳۴۵۶۷۸۹") == "0123456789"

    # Arabic-Indic digits ٠-٩ -> 0-9
    assert normalize_ar("٠١٢٣٤٥٦٧٨٩") == "0123456789"

    # Diacritics U+064B–U+0652
    assert normalize_ar("عَقْدٌ مَفْتُوحٌ") == "عقد مفتوح"
    assert normalize_ar("كِتَابًا مُبِينًا") == "كتابا مبينا"

    # Tatweel U+0640
    assert normalize_ar("عــــقـــــد") == "عقد"

    # Alef forms
    assert normalize_ar("أ إ آ ٱ") == "ا ا ا ا"

    # Alef maqsura ى -> ي
    assert normalize_ar("مستشفى ومرمى") == "مستشفي ومرمي"

    # Ta marbuta ة -> ه
    assert normalize_ar("شركة مساهمة") == "شركه مساهمه"

    # ؤ and ئ stay as they are!
    assert normalize_ar("مسؤول ورئيس ولؤلؤ") == "مسؤول ورئيس ولؤلؤ"

    # Mixed scripts & lowercase
    assert normalize_ar("  CONTRACT No. ١٢٣ / ۴۵۶  ") == "contract no. 123 / 456"


# 2. tokenize_mixed_scripts (4 pts)
def test_tokenize_mixed_scripts():
    tokens = tokenize("عقد رقم 123: sha7en iPhone, 3al-tareeq! هل تم؟")
    assert "عقد" in tokens
    assert "رقم" in tokens
    assert "123" in tokens
    assert "sha7en" in tokens
    assert "iphone" in tokens
    assert "3al" in tokens
    assert "tareeq" in tokens
    assert "هل" in tokens
    assert "تم" in tokens
    # Punctuation should not be in tokens
    assert ":" not in tokens
    assert "!" not in tokens
    assert "؟" not in tokens
    assert "," not in tokens


# 3 & 4. chunk_offsets_property (10 pts) and chunk_coverage_and_overlap_property (8 pts)
def test_chunk_properties_random_texts():
    alphabet = "abcdef ghijk \n\n .!? ؟۔ \n 1234567890 أحكام العقد والمسؤولية "
    rng = random.Random(42)
    for _ in range(300):
        length = rng.randint(1, 1500)
        text = "".join(rng.choices(alphabet, k=length))
        max_chars = rng.randint(20, 150)
        overlap = rng.randint(0, max_chars - 1)

        chunks = chunk_text("doc", text, max_chars=max_chars, overlap=overlap)

        if not text.strip():
            assert chunks == []
            continue

        # Rule 1 & 2:
        for i, c in enumerate(chunks):
            assert text[c.start:c.end] == c.text
            assert len(c.text) > 0
            assert not c.text[0].isspace()
            assert not c.text[-1].isspace()
            assert len(c.text) <= max_chars
            assert c.chunk_id == f"doc#{i}"
            if i > 0:
                assert c.start > chunks[i - 1].start

        # Rule 3: every non-whitespace character in text lies inside at least one chunk
        non_ws_indices = {idx for idx, ch in enumerate(text) if not ch.isspace()}
        covered = set()
        for c in chunks:
            covered.update(range(c.start, c.end))
        assert non_ws_indices.issubset(covered)

        # Rule 4: consecutive chunks overlap by at most overlap characters,
        # and no non-whitespace character lies between them
        for i in range(len(chunks) - 1):
            c1, c2 = chunks[i], chunks[i + 1]
            ov = max(0, c1.end - c2.start)
            assert ov <= overlap
            if c2.start > c1.end:
                between = text[c1.end:c2.start]
                assert all(ch.isspace() for ch in between)

        # Rule 6: a chunk may start mid-word only when previous chunk ended with mid-word cut
        for i in range(1, len(chunks)):
            prev, curr = chunks[i - 1], chunks[i]
            curr_starts_mid = curr.start > 0 and not text[curr.start - 1].isspace()
            prev_ends_mid = (
                prev.end < len(text)
                and not text[prev.end - 1].isspace()
                and not text[prev.end].isspace()
            )
            if curr_starts_mid:
                assert prev_ends_mid


# 5. chunk_prefers_structure (6 pts)
def test_chunk_prefers_structure():
    # Paragraph break over sentence end over space
    # In window of 60 chars, put space at 39, '.' at 44, and '\n\n' at 50..52
    t1 = "a" * 39 + " " + "b" * 4 + "." + "c" * 4 + "\n\n" + "d" * 30
    chunks1 = chunk_text("doc", t1, max_chars=60, overlap=10)
    assert chunks1[0].text.endswith("cccc")  # chosen just after \n\n and stripped

    # Sentence end over space
    t2 = "a" * 35 + "." + "b" * 10 + " " + "c" * 30
    chunks2 = chunk_text("doc", t2, max_chars=60, overlap=10)
    assert chunks2[0].text.endswith(".")  # chosen just after '.'

    # Sentence end with Arabic question mark and urdu full stop
    t3 = "a" * 35 + "؟" + "b" * 10 + " " + "c" * 30
    chunks3 = chunk_text("doc", t3, max_chars=60, overlap=10)
    assert chunks3[0].text.endswith("؟")

    t4 = "a" * 35 + "۔" + "b" * 10 + " " + "c" * 30
    chunks4 = chunk_text("doc", t4, max_chars=60, overlap=10)
    assert chunks4[0].text.endswith("۔")


# 6. chunk_edge_cases (5 pts)
def test_chunk_edge_cases():
    assert chunk_text("doc", "") == []
    assert chunk_text("doc", "   \t\n   \n\n  ") == []

    # One 2,000-character word
    word_2k = "a" * 2000
    chunks = chunk_text("doc", word_2k, max_chars=500, overlap=50)
    assert len(chunks) == 5
    for i, c in enumerate(chunks):
        assert word_2k[c.start:c.end] == c.text
        assert c.chunk_id == f"doc#{i}"
        assert len(c.text) <= 500

    # Bad arguments raise ValueError
    with pytest.raises(ValueError):
        chunk_text("doc", "text", max_chars=0, overlap=0)
    with pytest.raises(ValueError):
        chunk_text("doc", "text", max_chars=-10, overlap=0)
    with pytest.raises(ValueError):
        chunk_text("doc", "text", max_chars=50, overlap=-1)
    with pytest.raises(ValueError):
        chunk_text("doc", "text", max_chars=50, overlap=50)
    with pytest.raises(ValueError):
        chunk_text("doc", "text", max_chars=50, overlap=60)


# 7. bm25_matches_reference_scores (12 pts)
def test_bm25_matches_reference_scores():
    # 40-chunk synthetic corpus
    chunks = [
        Chunk(doc_id="d", chunk_id=f"d#{i}", start=0, end=10, text=f"clause {i} apple banana cherry {i%3}")
        for i in range(40)
    ]
    query = "clause apple 0"

    for k1, b in [(1.5, 0.75), (1.2, 0.5), (2.0, 0.8)]:
        index = BM25Index(k1=k1, b=b)
        index.add(chunks)
        results = index.search(query, top_k=40)

        # Manual computation
        N = 40
        q_tokens = tokenize(query)
        u_terms = set(q_tokens)
        doc_tokens = [tokenize(c.text) for c in chunks]
        doc_lens = [len(toks) for toks in doc_tokens]
        avgdl = sum(doc_lens) / N

        df = {}
        for toks in doc_tokens:
            for term in set(toks):
                df[term] = df.get(term, 0) + 1

        ref_scores = {}
        for i in range(N):
            toks = doc_tokens[i]
            dl = doc_lens[i]
            s = 0.0
            for t in u_terms:
                if t in df:
                    d_freq = df[t]
                    idf = math.log(1.0 + (N - d_freq + 0.5) / (d_freq + 0.5))
                    tf = toks.count(t)
                    if tf > 0:
                        denom = tf + k1 * (1.0 - b + b * (dl / avgdl))
                        s += idf * tf * (k1 + 1.0) / denom
            if s > 0:
                ref_scores[f"d#{i}"] = s

        ref_sorted = sorted(ref_scores.items(), key=lambda x: (-x[1], x[0]))

        assert len(results) == len(ref_sorted)
        for (res_id, res_s), (ref_id, ref_s) in zip(results, ref_sorted):
            assert res_id == ref_id
            assert abs(res_s - ref_s) < 1e-9


# 8. bm25_uses_normalisation (5 pts)
def test_bm25_uses_normalisation():
    chunks = [
        Chunk(doc_id="c", chunk_id="c#0", start=0, end=20, text="أَحْكَامُ الْعَقْدِ فِي البَنْدِ ١٢"),
    ]
    index = BM25Index()
    index.add(chunks)

    # Search with bare alef, without diacritics, ASCII digits
    res = index.search("احكام العقد في البند 12")
    assert len(res) == 1
    assert res[0][0] == "c#0"
    assert res[0][1] > 0


# 9. bm25_ordering_and_limits (5 pts)
def test_bm25_ordering_and_limits():
    chunks = [
        Chunk(doc_id="d", chunk_id="d#10", start=0, end=10, text="same score term"),
        Chunk(doc_id="d", chunk_id="d#2", start=0, end=10, text="same score term"),
        Chunk(doc_id="d", chunk_id="d#1", start=0, end=10, text="same score term"),
        Chunk(doc_id="d", chunk_id="d#3", start=0, end=10, text="different unmatching"),
    ]
    index = BM25Index()
    index.add(chunks)

    # Repeated query terms count once
    res1 = index.search("term term term", top_k=2)
    res2 = index.search("term", top_k=2)
    assert res1 == res2

    # Tie break by chunk ID ascending: string order: 'd#1' < 'd#10' < 'd#2'
    res_all = index.search("term", top_k=10)
    assert len(res_all) == 3
    assert [cid for cid, _ in res_all] == ["d#1", "d#10", "d#2"]

    # Empty query returns []
    assert index.search("") == []
    assert index.search("   ") == []

    # top_k limit
    assert len(index.search("term", top_k=1)) == 1


# 10. rrf_correctness (7 pts)
def test_rrf_correctness():
    # Items missing from one list
    rankings = [["a", "b", "c"], ["b", "d"]]
    fused = rrf(rankings, k=60)
    fused_dict = dict(fused)
    # b is rank 2 in list 0 (1/62) and rank 1 in list 1 (1/61)
    assert fused_dict["b"] == pytest.approx(1 / 62 + 1 / 61)
    # a is rank 1 in list 0 only (1/61)
    assert fused_dict["a"] == pytest.approx(1 / 61)
    # d is rank 2 in list 1 only (1/62)
    assert fused_dict["d"] == pytest.approx(1 / 62)

    # Duplicates in single list should count at best rank
    dup_rankings = [["a", "a", "b"], ["b"]]
    fused_dup = rrf(dup_rankings, k=60)
    assert dict(fused_dup)["a"] == pytest.approx(1 / 61)

    # Custom k
    fused_k = rrf([["x"], ["x"]], k=10)
    assert fused_k[0][1] == pytest.approx(2 / 11)

    # Tie order by ID ascending
    tie_rankings = [["b", "a"]]  # b has rank 1 (1/61), a has rank 2 (1/62)
    tie_rankings2 = [["x", "y"], ["y", "x"]]  # both x and y have 1/61 + 1/62
    fused_ties = rrf(tie_rankings2, k=60)
    assert fused_ties[0][0] == "x"
    assert fused_ties[1][0] == "y"


# 11. hybrid_uses_dense_path (8 pts)
class MockDenseOnlyEmbedder:
    """Embedder where chunk0 has high dot product with query, while BM25 has no lexical match."""
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        res = []
        for t in texts:
            if "semantic_match" in t or "query_vector" in t:
                res.append(np.array([1.0, 0.0]))
            else:
                res.append(np.array([0.0, 1.0]))
        return np.array(res)


def test_hybrid_uses_dense_path():
    chunks = [
        Chunk(doc_id="d", chunk_id="d#1", start=0, end=14, text="semantic_match"),
        Chunk(doc_id="d", chunk_id="d#2", start=0, end=14, text="other text here"),
    ]
    retriever = HybridRetriever(chunks, MockDenseOnlyEmbedder())
    # Query has no lexical overlap with semantic_match
    hits = retriever.search("query_vector", top_k=1)
    assert len(hits) == 1
    assert hits[0].chunk_id == "d#1"


# 12. hybrid_hits_carry_offsets (5 pts)
def test_hybrid_hits_carry_offsets():
    contract = "Line 1 paragraph here.\n\nLine 2 another paragraph here."
    chunks = chunk_text("doc1", contract, max_chars=40, overlap=5)
    retriever = HybridRetriever(chunks, HashingEmbedder())
    hits = retriever.search("paragraph", top_k=2)
    for h in hits:
        assert contract[h.start:h.end] == h.text
        # Score is RRF score > 0
        assert h.score > 0


# 13. reranker_hook (5 pts)
def test_reranker_hook():
    chunks = [
        Chunk(doc_id="d", chunk_id="d#1", start=0, end=10, text="apple pie"),
        Chunk(doc_id="d", chunk_id="d#2", start=0, end=10, text="banana split"),
    ]
    call_count = 0

    def mock_reranker(query: str, candidates: list[Hit]) -> list[float]:
        nonlocal call_count
        call_count += 1
        # Give d#2 highest score
        return [10.0 if h.chunk_id == "d#2" else 1.0 for h in candidates]

    retriever = HybridRetriever(chunks, HashingEmbedder(), reranker=mock_reranker)
    hits = retriever.search("apple", top_k=2)

    assert call_count == 1
    assert hits[0].chunk_id == "d#2"
    assert hits[0].score == 10.0
    assert hits[1].chunk_id == "d#1"
    assert hits[1].score == 1.0


# 14. citation_checks (4 pts)
def test_citation_checks():
    doc = "This is the full contract text. It contains terms and conditions."
    assert verify_citation("full contract text", doc, 12, 31)
    # Whitespace collapsed match
    assert verify_citation("full   contract   text", doc, 12, 31)
    # Blank quote
    assert not verify_citation("", doc, 12, 31)
    assert not verify_citation("   ", doc, 12, 31)
    # Bad start/end
    assert not verify_citation("full", doc, -1, 10)
    assert not verify_citation("full", doc, 20, 10)
    assert not verify_citation("full", doc, 10, len(doc) + 5)
    # Quote not in slice
    assert not verify_citation("unrelated text", doc, 0, 20)

    # format_citation truncation
    hit = Hit(
        chunk_id="d#0",
        doc_id="doc1",
        start=5,
        end=200,
        text="word1 word2 word3 word4 word5 word6 word7 word8 word9 word10",
        score=0.95,
    )
    formatted = format_citation(hit, max_quote=25)
    assert formatted.startswith('[doc1:5-200] "')
    assert formatted.endswith('…"')
    # Check that it cut back to last space inside first 25 chars
    # text[:25] is "word1 word2 word3 word4 w"
    # last space is at index 23 (after word4)
    # so prefix is "word1 word2 word3 word4"
    assert formatted == '[doc1:5-200] "word1 word2 word3 word4…"'
