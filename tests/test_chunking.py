"""
test_chunking.py — pytest tests for all three chunking strategies.
"""
from __future__ import annotations

import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chunking import (
    Chunk,
    ChunkMetadata,
    chunk_fixed,
    chunk_metadata_aware,
    chunk_passage,
    chunk_sentence,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_PASSAGE = {
    "passage_id": "test_001",
    "text": (
        "Machine learning is a subset of artificial intelligence. "
        "It enables computers to learn from data without being explicitly programmed. "
        "Deep learning uses neural networks with many layers. "
        "These networks can automatically learn features from raw data. "
        "Applications include image recognition, natural language processing, and more. "
        "Reinforcement learning trains agents through rewards and penalties. "
        "Transfer learning reuses models trained on one task for another. "
        "Data quality and quantity are critical factors in model performance."
    ),
    "query": "What is machine learning?",
}

SHORT_PASSAGE = {
    "passage_id": "short_001",
    "text": "Short text.",
    "query": None,
}


# ── Strategy 1: Fixed-size chunking ──────────────────────────────────────────

class TestFixedChunking:
    def test_returns_nonempty_chunks(self):
        chunks = chunk_fixed(SAMPLE_PASSAGE, chunk_size=50, overlap_pct=0.2)
        assert len(chunks) > 0

    def test_chunk_schema(self):
        chunks = chunk_fixed(SAMPLE_PASSAGE, chunk_size=50)
        for chunk in chunks:
            assert "text" in chunk
            assert "metadata" in chunk
            meta = chunk["metadata"]
            assert meta["passage_id"] == "test_001"
            assert meta["strategy_used"] == "fixed"
            assert isinstance(meta["position_in_doc"], int)
            assert isinstance(meta["chunk_id"], str)

    def test_chunk_text_nonempty(self):
        chunks = chunk_fixed(SAMPLE_PASSAGE, chunk_size=50)
        for chunk in chunks:
            assert len(chunk["text"].strip()) > 0

    def test_overlap_increases_chunk_count(self):
        chunks_no_overlap = chunk_fixed(SAMPLE_PASSAGE, chunk_size=50, overlap_pct=0.0)
        chunks_with_overlap = chunk_fixed(SAMPLE_PASSAGE, chunk_size=50, overlap_pct=0.5)
        assert len(chunks_with_overlap) >= len(chunks_no_overlap)

    def test_chunk_size_respected(self):
        """Each chunk should be ≤ chunk_size + small tolerance (overlap causes slight overage)."""
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        chunk_size = 30
        chunks = chunk_fixed(SAMPLE_PASSAGE, chunk_size=chunk_size, overlap_pct=0.0)
        for chunk in chunks:
            token_count = len(enc.encode(chunk["text"]))
            assert token_count <= chunk_size + 5, f"Chunk too large: {token_count} tokens"

    def test_short_passage_produces_single_chunk(self):
        chunks = chunk_fixed(SHORT_PASSAGE, chunk_size=256)
        assert len(chunks) == 1

    def test_positions_are_sequential(self):
        chunks = chunk_fixed(SAMPLE_PASSAGE, chunk_size=50, overlap_pct=0.0)
        positions = [c["metadata"]["position_in_doc"] for c in chunks]
        assert positions == list(range(len(chunks)))


# ── Strategy 2: Sentence chunking ────────────────────────────────────────────

class TestSentenceChunking:
    def test_returns_nonempty_chunks(self):
        chunks = chunk_sentence(SAMPLE_PASSAGE, chunk_size=80)
        assert len(chunks) > 0

    def test_chunk_schema(self):
        chunks = chunk_sentence(SAMPLE_PASSAGE, chunk_size=80)
        for chunk in chunks:
            assert "text" in chunk
            assert chunk["metadata"]["strategy_used"] == "sentence"
            assert chunk["metadata"]["passage_id"] == "test_001"

    def test_chunks_dont_exceed_budget(self):
        """Chunks should not significantly exceed chunk_size (sentence boundary may push slightly over)."""
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        chunk_size = 60
        chunks = chunk_sentence(SAMPLE_PASSAGE, chunk_size=chunk_size)
        for chunk in chunks:
            # A single long sentence may exceed budget — that's acceptable
            sentences_count = len(chunk["text"].split(". "))
            if sentences_count > 1:
                token_count = len(enc.encode(chunk["text"]))
                assert token_count <= chunk_size * 1.5, f"Multi-sentence chunk too large: {token_count}"

    def test_source_query_preserved(self):
        chunks = chunk_sentence(SAMPLE_PASSAGE)
        for chunk in chunks:
            assert chunk["metadata"]["source_query"] == "What is machine learning?"

    def test_no_query_passage(self):
        passage = {**SAMPLE_PASSAGE, "query": None}
        chunks = chunk_sentence(passage)
        for chunk in chunks:
            assert chunk["metadata"]["source_query"] is None


# ── Strategy 3: Metadata-aware chunking ──────────────────────────────────────

class TestMetadataChunking:
    def test_returns_nonempty_chunks(self):
        chunks = chunk_metadata_aware(SAMPLE_PASSAGE, chunk_size=50)
        assert len(chunks) > 0

    def test_all_metadata_fields_populated(self):
        chunks = chunk_metadata_aware(SAMPLE_PASSAGE, chunk_size=50)
        required_fields = {"passage_id", "chunk_id", "source_query", "position_in_doc", "strategy_used"}
        for chunk in chunks:
            assert required_fields == set(chunk["metadata"].keys())
            assert chunk["metadata"]["strategy_used"] == "metadata"

    def test_chunk_ids_unique(self):
        chunks = chunk_metadata_aware(SAMPLE_PASSAGE, chunk_size=50)
        chunk_ids = [c["metadata"]["chunk_id"] for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids)), "Chunk IDs must be unique"

    def test_source_query_always_string(self):
        """Metadata strategy should coerce None query to empty string."""
        passage = {**SAMPLE_PASSAGE, "query": None}
        chunks = chunk_metadata_aware(passage, chunk_size=50)
        for chunk in chunks:
            assert isinstance(chunk["metadata"]["source_query"], str)

    def test_position_in_doc_sequential(self):
        chunks = chunk_metadata_aware(SAMPLE_PASSAGE, chunk_size=50, overlap_pct=0.0)
        positions = [c["metadata"]["position_in_doc"] for c in chunks]
        assert positions == list(range(len(chunks)))


# ── Dispatcher ────────────────────────────────────────────────────────────────

class TestDispatcher:
    @pytest.mark.parametrize("strategy", ["fixed", "sentence", "metadata"])
    def test_all_strategies_via_dispatcher(self, strategy: str):
        chunks = chunk_passage(SAMPLE_PASSAGE, strategy=strategy, chunk_size=80)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk["metadata"]["strategy_used"] == strategy

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            chunk_passage(SAMPLE_PASSAGE, strategy="magic_chunker")

    def test_consistent_schema_across_strategies(self):
        """All strategies must return the same top-level keys."""
        required = {"text", "metadata"}
        meta_required = {"passage_id", "chunk_id", "source_query", "position_in_doc", "strategy_used"}
        for strategy in ["fixed", "sentence", "metadata"]:
            chunks = chunk_passage(SAMPLE_PASSAGE, strategy=strategy, chunk_size=100)
            for chunk in chunks:
                assert set(chunk.keys()) == required
                assert set(chunk["metadata"].keys()) == meta_required
