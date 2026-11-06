import numpy as np
import pytest

from bayyina.bm25 import BM25Index
from bayyina.chunking import chunk_text
from bayyina.cite import verify_citation
from bayyina.embedder import HashingEmbedder
from bayyina.hybrid import HybridRetriever, rrf
from bayyina.normalize import normalize_ar, tokenize

CONTRACT = ("المادة 5: مدة العقد\nمدة هذا العقد سنة واحدة تبدأ من تاريخ التوقيع.\n\n"
            "المادة 9: حدود المسئولية\nلا تتجاوز مسئولية المورد قيمة الطلبية محل النزاع.\n\n"
            "Clause 11 Governing law. This agreement is governed by the laws of Egypt.")


def test_normalize_handles_diacritics_tatweel_alef_and_digits():
    assert normalize_ar("إِنَّ الـــعقدَ في ٣١") == "ان العقد في 31"
    assert tokenize("sha7en iPhone شاحن") == ["sha7en", "iphone", "شاحن"]


def test_chunks_keep_exact_offsets():
    chunks = chunk_text("supply_01", CONTRACT, max_chars=80, overlap=10)
    assert chunks and chunks[0].chunk_id == "supply_01#0"
    for c in chunks:
        assert CONTRACT[c.start:c.end] == c.text
        assert len(c.text) <= 80


def test_bm25_finds_the_liability_clause_despite_spelling():
    chunks = chunk_text("supply_01", CONTRACT, max_chars=80, overlap=10)
    index = BM25Index()
    index.add(chunks)
    best_id, _ = index.search("مسؤولية المورد", top_k=1)[0]   # note: ؤ not ئ in the query
    best = next(c for c in chunks if c.chunk_id == best_id)
    assert "المورد" in best.text


def test_rrf_rewards_agreement():
    fused = rrf([["a", "b", "c"], ["c", "d"]], k=60)
    assert fused[0][0] == "c"
    assert fused[0][1] == pytest.approx(1 / 63 + 1 / 61)


def test_hybrid_hits_cite_the_original_text():
    chunks = chunk_text("supply_01", CONTRACT, max_chars=80, overlap=10)
    hits = HybridRetriever(chunks, HashingEmbedder()).search("governing law", top_k=2)
    assert hits and hits[0].doc_id == "supply_01"
    h = hits[0]
    assert verify_citation(h.text, CONTRACT, h.start, h.end)
    assert not verify_citation("governed by the laws of France", CONTRACT, h.start, h.end)
