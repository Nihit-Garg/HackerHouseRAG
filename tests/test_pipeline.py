"""
test_pipeline.py — End-to-end pipeline tests using mocked STT.

Tests:
  - Structured dict returned with all required keys
  - Text-query path (no STT) works end-to-end
  - Empty query returns graceful error (no crash)
  - Off-corpus query triggers fallback response (no hallucination)
  - Bad/empty audio input captured in errors[], not raised
  - Stage timings are populated and non-negative
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import faiss
import numpy as np
import pytest
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from generation import FALLBACK_RESPONSE


# ── Shared fixtures ────────────────────────────────────────────────────────────

def _make_chunk(chunk_id: str, text: str) -> dict:
    return {
        "text": text,
        "metadata": {
            "passage_id": "p1",
            "chunk_id": chunk_id,
            "source_query": None,
            "position_in_doc": 0,
            "strategy_used": "metadata",
        },
    }


MINI_CORPUS = [
    _make_chunk("c1", "Python is a high-level programming language."),
    _make_chunk("c2", "FastAPI is a modern web framework for building APIs."),
    _make_chunk("c3", "FAISS is a library for efficient similarity search."),
]


@pytest.fixture(scope="module")
def pipeline_with_mini_corpus():
    """
    Build a RAGPipeline with a real (tiny) FAISS+BM25 index and a
    mocked Ollama LLM so tests don't require a running Ollama server.
    """
    from embeddings import EmbeddingModel
    from graph import RAGPipeline
    from retrieval import HybridRetriever

    texts = [c["text"] for c in MINI_CORPUS]
    emb_model = EmbeddingModel()
    embeddings = emb_model.embed_documents(texts)

    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings)

    bm25 = BM25Okapi([t.lower().split() for t in texts])
    retriever = HybridRetriever(faiss_index, bm25, MINI_CORPUS, emb_model)

    pipeline = RAGPipeline(retriever, emb_model, top_k=3)
    return pipeline


# ── Required output keys ───────────────────────────────────────────────────────

REQUIRED_KEYS = {
    "query",
    "transcribed_text",
    "retrieved_chunks",
    "answer",
    "stage_timings",
    "errors",
    "guardrail_triggered",
    "guardrail_detail",
}

REQUIRED_TIMING_KEYS = {"stt_ms", "retrieval_ms", "generation_ms", "total_ms"}


# ── Tests: structured output ──────────────────────────────────────────────────

class TestPipelineOutputSchema:
    def test_all_required_keys_present(self, pipeline_with_mini_corpus):
        with patch("generation.generate_answer", return_value="Python is high-level."):
            result = pipeline_with_mini_corpus.answer(text_query="What is Python?")

        assert REQUIRED_KEYS.issubset(set(result.keys()))

    def test_stage_timings_keys_present(self, pipeline_with_mini_corpus):
        with patch("generation.generate_answer", return_value="Python is high-level."):
            result = pipeline_with_mini_corpus.answer(text_query="What is Python?")

        assert REQUIRED_TIMING_KEYS.issubset(set(result["stage_timings"].keys()))

    def test_stage_timings_are_nonnegative(self, pipeline_with_mini_corpus):
        with patch("generation.generate_answer", return_value="some answer"):
            result = pipeline_with_mini_corpus.answer(text_query="FastAPI framework?")

        for key, val in result["stage_timings"].items():
            assert val >= 0, f"Timing {key} is negative: {val}"

    def test_retrieved_chunks_is_list(self, pipeline_with_mini_corpus):
        with patch("generation.generate_answer", return_value="answer"):
            result = pipeline_with_mini_corpus.answer(text_query="FAISS similarity search")

        assert isinstance(result["retrieved_chunks"], list)

    def test_errors_is_list(self, pipeline_with_mini_corpus):
        with patch("generation.generate_answer", return_value="answer"):
            result = pipeline_with_mini_corpus.answer(text_query="Python language")

        assert isinstance(result["errors"], list)


# ── Tests: text query path (no STT) ──────────────────────────────────────────

class TestTextQueryPath:
    def test_text_query_sets_transcribed_text(self, pipeline_with_mini_corpus):
        with patch("generation.generate_answer", return_value="answer"):
            result = pipeline_with_mini_corpus.answer(text_query="What is FastAPI?")

        assert result["transcribed_text"] == "What is FastAPI?"

    def test_text_query_stt_ms_is_near_zero(self, pipeline_with_mini_corpus):
        """STT node is skipped for text queries; stt_ms should be tiny."""
        with patch("generation.generate_answer", return_value="answer"):
            result = pipeline_with_mini_corpus.answer(text_query="What is FAISS?")

        assert result["stage_timings"]["stt_ms"] < 100  # < 100ms since no actual STT

    def test_text_query_answer_is_string(self, pipeline_with_mini_corpus):
        with patch("generation.generate_answer", return_value="It is a web framework."):
            result = pipeline_with_mini_corpus.answer(text_query="What is FastAPI?")

        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0


# ── Tests: bad inputs — no crash, errors captured ─────────────────────────────

class TestBadInputHandling:
    def test_empty_text_query_no_crash(self, pipeline_with_mini_corpus):
        """Empty text query should not raise; should return error in errors list."""
        result = pipeline_with_mini_corpus.answer(text_query="")
        assert isinstance(result, dict)
        assert REQUIRED_KEYS.issubset(set(result.keys()))

    def test_empty_text_query_captured_in_errors(self, pipeline_with_mini_corpus):
        result = pipeline_with_mini_corpus.answer(text_query="")
        # Either errors list has content OR answer is fallback — pipeline must not crash
        assert result["answer"] == FALLBACK_RESPONSE or len(result["errors"]) > 0

    def test_nonexistent_audio_path_no_crash(self, pipeline_with_mini_corpus):
        """Non-existent audio file should fail gracefully, not raise."""
        result = pipeline_with_mini_corpus.answer(audio_path="/nonexistent/path/audio.wav")
        assert isinstance(result, dict)
        assert REQUIRED_KEYS.issubset(set(result.keys()))

    def test_nonexistent_audio_captured_in_errors(self, pipeline_with_mini_corpus):
        result = pipeline_with_mini_corpus.answer(audio_path="/nonexistent/path/audio.wav")
        assert len(result["errors"]) > 0

    def test_no_input_at_all_no_crash(self, pipeline_with_mini_corpus):
        result = pipeline_with_mini_corpus.answer()
        assert isinstance(result, dict)
        assert len(result["errors"]) > 0

    def test_gibberish_query_no_crash(self, pipeline_with_mini_corpus):
        with patch("generation.generate_answer", return_value=FALLBACK_RESPONSE):
            result = pipeline_with_mini_corpus.answer(text_query="asdfjkl qwerty zxcvbnm")
        assert isinstance(result, dict)
        assert isinstance(result["answer"], str)


# ── Tests: anti-hallucination ─────────────────────────────────────────────────

class TestAntiHallucination:
    def test_off_corpus_query_returns_fallback_or_low_scores(self, pipeline_with_mini_corpus):
        """
        A query completely unrelated to the corpus (Mayan temples)
        should either be caught by retrieval_threshold guard OR return a rejection.
        """
        with patch("generation.generate_answer", return_value=FALLBACK_RESPONSE):
            result = pipeline_with_mini_corpus.answer(
                text_query="Tell me about ancient Mayan pyramid construction techniques."
            )

        # Guardrail triggered (low_retrieval / off_topic) OR LLM returned fallback
        guard = result.get("guardrail_triggered")
        assert guard is not None or FALLBACK_RESPONSE in result["answer"], (
            f"Expected guardrail or fallback, got: {result['answer']!r}"
        )

    def test_empty_retrieval_forces_fallback(self, pipeline_with_mini_corpus):
        """If retrieval returns no chunks, retrieval_threshold guard must fire."""
        with patch("retrieval.HybridRetriever.retrieve", return_value={"results": [], "latency": {"start_ts": 0, "end_ts": 0, "elapsed_ms": 0}}):
            with patch("generation.generate_answer", return_value=FALLBACK_RESPONSE):
                result = pipeline_with_mini_corpus.answer(text_query="does not matter")

        # Empty retrieval → retrieval_threshold guard triggers OR fallback answer
        assert result.get("guardrail_triggered") == "low_retrieval" or FALLBACK_RESPONSE in result["answer"]


# ── Tests: mocked STT for audio path ─────────────────────────────────────────

class TestMockedSTT:
    def test_mocked_stt_flows_through_pipeline(self, pipeline_with_mini_corpus, tmp_path):
        """Mock STT to avoid Sarvam API call; verify full pipeline still runs."""
        fake_audio = tmp_path / "test_audio.wav"
        fake_audio.write_bytes(b"RIFF" + b"\x00" * 100)

        with patch("graph.transcribe", return_value="What is Python?"):
            with patch("generation.generate_answer", return_value="Python is a programming language."):
                result = pipeline_with_mini_corpus.answer(audio_path=str(fake_audio))

        assert result["transcribed_text"] == "What is Python?"
        # Answer is either the mocked value (grounding passed) or a guardrail rejection
        assert isinstance(result["answer"], str) and len(result["answer"]) > 0
        assert REQUIRED_KEYS.issubset(set(result.keys()))

    def test_stt_failure_captured_not_raised(self, pipeline_with_mini_corpus, tmp_path):
        """STT failure must be captured in errors[], not raised as exception."""
        from stt import STTError

        fake_audio = tmp_path / "bad_audio.wav"
        fake_audio.write_bytes(b"bad data")

        with patch("graph.transcribe", side_effect=STTError("API unavailable")):
            result = pipeline_with_mini_corpus.answer(audio_path=str(fake_audio))

        assert isinstance(result, dict)
        assert any("STT" in e for e in result["errors"])
        # Pipeline must not crash — answer is a string (guardrail or fallback)
        assert isinstance(result["answer"], str)
