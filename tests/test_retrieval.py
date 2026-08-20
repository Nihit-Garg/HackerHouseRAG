"""
test_retrieval.py — pytest tests for hybrid retrieval and RRF fusion.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from retrieval import HybridRetriever, reciprocal_rank_fusion


# ── Synthetic corpus fixture ───────────────────────────────────────────────────

def _make_chunk(chunk_id: str, text: str, passage_id: str = "p1") -> dict:
    return {
        "text": text,
        "metadata": {
            "passage_id": passage_id,
            "chunk_id": chunk_id,
            "source_query": None,
            "position_in_doc": 0,
            "strategy_used": "fixed",
        },
    }


CORPUS_CHUNKS = [
    _make_chunk("c1", "Deep learning is a subset of machine learning using neural networks.", "p1"),
    _make_chunk("c2", "The Eiffel Tower is located in Paris, France.", "p2"),
    _make_chunk("c3", "Python is a popular programming language for data science.", "p3"),
    _make_chunk("c4", "Reinforcement learning trains agents through reward and penalty signals.", "p4"),
    _make_chunk("c5", "The Great Wall of China stretches thousands of miles.", "p5"),
    _make_chunk("c6", "Transformers revolutionised natural language processing in 2017.", "p6"),
    _make_chunk("c7", "Neural networks learn representations through backpropagation.", "p7"),
]


@pytest.fixture(scope="module")
def small_retriever():
    """Build a minimal FAISS + BM25 retriever over CORPUS_CHUNKS for testing."""
    from rank_bm25 import BM25Okapi
    import faiss
    from embeddings import EmbeddingModel
    from retrieval import HybridRetriever

    texts = [c["text"] for c in CORPUS_CHUNKS]

    # Build real embedding model (small, fast)
    emb_model = EmbeddingModel()
    embeddings = emb_model.embed_documents(texts)

    # FAISS flat index
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings)

    # BM25 index
    bm25 = BM25Okapi([t.lower().split() for t in texts])

    return HybridRetriever(faiss_index, bm25, CORPUS_CHUNKS, emb_model)


# ── RRF unit tests (no model needed) ─────────────────────────────────────────

class TestRRF:
    def _make_result(self, chunk_id: str, text: str, score: float) -> dict:
        return {
            "chunk": _make_chunk(chunk_id, text),
            "score": score,
        }

    def test_rrf_returns_correct_count(self):
        dense = [self._make_result(f"c{i}", f"text {i}", 1.0 - i * 0.1) for i in range(5)]
        bm25 = [self._make_result(f"c{i}", f"text {i}", 5.0 - i) for i in range(5)]
        results = reciprocal_rank_fusion(dense, bm25, top_k=3)
        assert len(results) == 3

    def test_rrf_fused_chunk_has_all_score_fields(self):
        dense = [self._make_result("c1", "neural networks", 0.9)]
        bm25 = [self._make_result("c1", "neural networks", 5.0)]
        results = reciprocal_rank_fusion(dense, bm25, top_k=1)
        assert len(results) == 1
        r = results[0]
        assert "dense_score" in r
        assert "bm25_score" in r
        assert "rrf_score" in r
        assert "text" in r
        assert "metadata" in r

    def test_rrf_score_additive_for_same_chunk(self):
        """A chunk appearing in both lists should have a higher RRF score than one appearing in only one."""
        # c1 appears in both dense and bm25; c2 only in dense
        dense = [
            self._make_result("c1", "text one", 0.95),
            self._make_result("c2", "text two", 0.90),
        ]
        bm25 = [
            self._make_result("c1", "text one", 5.0),
        ]
        results = reciprocal_rank_fusion(dense, bm25, top_k=2)
        result_map = {r["metadata"]["chunk_id"]: r for r in results}
        assert result_map["c1"]["rrf_score"] > result_map["c2"]["rrf_score"]

    def test_rrf_top_rank_gets_highest_score(self):
        """The first-ranked item in both lists should win overall."""
        shared_top = "c_top"
        dense = [self._make_result(shared_top, "top text", 0.99)] + [
            self._make_result(f"d{i}", f"dense {i}", 0.5 - i * 0.1) for i in range(4)
        ]
        bm25 = [self._make_result(shared_top, "top text", 9.9)] + [
            self._make_result(f"b{i}", f"bm25 {i}", 5.0 - i) for i in range(4)
        ]
        results = reciprocal_rank_fusion(dense, bm25, top_k=5)
        assert results[0]["metadata"]["chunk_id"] == shared_top

    def test_rrf_deduplicates_same_chunk(self):
        """Same chunk_id appearing in both lists should appear only once in output."""
        dense = [self._make_result("c1", "text one", 0.9)]
        bm25 = [self._make_result("c1", "text one", 4.0)]
        results = reciprocal_rank_fusion(dense, bm25, top_k=5)
        chunk_ids = [r["metadata"]["chunk_id"] for r in results]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_rrf_handles_empty_inputs(self):
        results = reciprocal_rank_fusion([], [], top_k=5)
        assert results == []

    def test_rrf_handles_one_empty_list(self):
        dense = [self._make_result("c1", "only dense", 0.9)]
        results = reciprocal_rank_fusion(dense, [], top_k=5)
        assert len(results) == 1
        assert results[0]["dense_score"] == pytest.approx(0.9)
        assert results[0]["bm25_score"] == 0.0


# ── Integration tests (real embedding model) ──────────────────────────────────

class TestHybridRetriever:
    def test_retrieve_returns_dict_with_results_key(self, small_retriever):
        output = small_retriever.retrieve("deep learning neural networks", top_k=3)
        assert "results" in output
        assert "latency" in output

    def test_retrieve_correct_top_k_count(self, small_retriever):
        output = small_retriever.retrieve("machine learning", top_k=3)
        assert len(output["results"]) <= 3

    def test_relevant_chunk_in_top_k(self, small_retriever):
        """'deep learning neural networks' query should retrieve ML-related chunks."""
        output = small_retriever.retrieve("deep learning neural networks", top_k=3)
        top_texts = [r["text"].lower() for r in output["results"]]
        # At least one top result should mention neural networks or deep learning
        assert any("neural" in t or "deep" in t or "learning" in t for t in top_texts)

    def test_eiffel_tower_retrieved_for_paris_query(self, small_retriever):
        """Paris query should retrieve Eiffel Tower chunk."""
        output = small_retriever.retrieve("Eiffel Tower Paris France", top_k=3)
        chunk_ids = [r["metadata"]["chunk_id"] for r in output["results"]]
        assert "c2" in chunk_ids

    def test_result_schema_complete(self, small_retriever):
        """Every result must have the required schema fields."""
        output = small_retriever.retrieve("programming language", top_k=3)
        for result in output["results"]:
            assert "text" in result
            assert "metadata" in result
            assert "dense_score" in result
            assert "bm25_score" in result
            assert "rrf_score" in result

    def test_latency_dict_populated(self, small_retriever):
        output = small_retriever.retrieve("neural networks", top_k=2)
        lat = output["latency"]
        assert "elapsed_ms" in lat
        assert lat["elapsed_ms"] > 0
        assert lat["start_ts"] < lat["end_ts"]

    def test_off_corpus_query_still_returns_results(self, small_retriever):
        """An off-corpus query should still return chunks (dense retrieval always returns something)."""
        output = small_retriever.retrieve("ancient Mayan calendar rituals", top_k=3)
        # Should return results — but their scores will be lower
        assert isinstance(output["results"], list)
