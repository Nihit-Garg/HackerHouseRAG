"""
harness.py — Structured, defensive stage wrappers for the RAG pipeline.

Each stage function wraps one pipeline component with:
  - Input validation
  - try/except with configurable retries + exponential backoff
  - Structured StageResult output (never raw strings)
  - Per-stage timing in milliseconds

Usage:
    from harness import stt_stage, retrieval_stage, generation_stage

    result = retrieval_stage(query="what is machine learning?", retriever=retriever)
    if result["ok"]:
        chunks = result["output"]
    else:
        print(result["error"])
"""
from __future__ import annotations

import logging
import time
from typing import Any

from config import (
    GENERATION_MAX_RETRIES,
    GENERATION_RETRY_BASE_DELAY_S,
    STT_MAX_RETRIES,
    STT_RETRY_BASE_DELAY_S,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ok(stage: str, output: Any, elapsed_ms: float, retries_used: int = 0) -> dict[str, Any]:
    return {
        "ok": True,
        "stage": stage,
        "output": output,
        "elapsed_ms": round(elapsed_ms, 2),
        "retries_used": retries_used,
        "error": None,
    }


def _fail(stage: str, error: str, elapsed_ms: float, retries_used: int = 0) -> dict[str, Any]:
    return {
        "ok": False,
        "stage": stage,
        "output": None,
        "elapsed_ms": round(elapsed_ms, 2),
        "retries_used": retries_used,
        "error": error,
    }


def _retry_loop(
    fn,
    args: tuple,
    kwargs: dict,
    max_retries: int,
    base_delay: float,
    stage: str,
) -> tuple[Any, int, str | None]:
    """
    Call fn(*args, **kwargs) up to max_retries+1 times with exponential backoff.

    Returns:
        (result, retries_used, last_error_str_or_None)
    """
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            result = fn(*args, **kwargs)
            return result, attempt, None
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "%s attempt %d/%d failed: %s — retrying in %.1fs",
                    stage, attempt + 1, max_retries + 1, exc, delay,
                )
                time.sleep(delay)
            else:
                logger.error("%s failed after %d attempts: %s", stage, attempt + 1, exc)
    return None, max_retries, str(last_error)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — STT
# ─────────────────────────────────────────────────────────────────────────────

def stt_stage(
    audio_path: str,
    *,
    max_retries: int = STT_MAX_RETRIES,
    base_delay: float = STT_RETRY_BASE_DELAY_S,
) -> dict[str, Any]:
    """
    Transcribe an audio file using Sarvam STT.

    Input validation:
        - audio_path must be a non-empty string
        - File must exist on disk

    Returns:
        StageResult with output = transcribed text string.
    """
    import os
    from stt import STTError, transcribe

    t0 = time.perf_counter()

    # ── Input validation ──────────────────────────────────────────────────────
    if not audio_path or not audio_path.strip():
        return _fail("stt", "audio_path is empty.", 0.0)
    if not os.path.isfile(audio_path):
        return _fail("stt", f"Audio file not found: {audio_path}", 0.0)

    # ── Transcribe with retry ─────────────────────────────────────────────────
    result, retries, err = _retry_loop(
        transcribe, (audio_path,), {},
        max_retries=max_retries,
        base_delay=base_delay,
        stage="stt",
    )
    elapsed = (time.perf_counter() - t0) * 1000

    if err:
        return _fail("stt", err, elapsed, retries)

    transcribed = (result or "").strip()
    if not transcribed:
        return _fail("stt", "STT returned empty transcription.", elapsed, retries)

    return _ok("stt", transcribed, elapsed, retries)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def retrieval_stage(
    query: str,
    retriever: Any,
    top_k: int | None = None,
) -> dict[str, Any]:
    """
    Run hybrid dense+BM25 retrieval for a text query.

    Input validation:
        - query must be a non-empty string of >= 2 words

    Retrieval doesn't retry (it's deterministic + fast).

    Returns:
        StageResult with output = {results: list[chunk], latency: dict}.
    """
    t0 = time.perf_counter()

    # ── Input validation ──────────────────────────────────────────────────────
    if not query or not query.strip():
        return _fail("retrieval", "Query is empty.", 0.0)
    if len(query.split()) < 2:
        return _fail("retrieval", f"Query too short ('{query}') — need at least 2 words.", 0.0)

    # ── Retrieve ──────────────────────────────────────────────────────────────
    try:
        kwargs: dict[str, Any] = {}
        if top_k is not None:
            kwargs["top_k"] = top_k
        result = retriever.retrieve(query, **kwargs)
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        logger.error("Retrieval failed: %s", exc)
        return _fail("retrieval", str(exc), elapsed)

    elapsed = (time.perf_counter() - t0) * 1000
    return _ok("retrieval", result, elapsed)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Generation
# ─────────────────────────────────────────────────────────────────────────────

# Grounding-aware system prompt — asks LLM to append a JSON grounding block.
_GROUNDED_SYSTEM_PROMPT = """\
You are a precise question-answering assistant.
Answer ONLY using the provided context passages. Do not use outside knowledge.
If the context does not contain enough information, respond with exactly:
"I don't have enough information in the provided context to answer that."

After your answer, on a new line, append a JSON object:
{"grounded": true, "confidence": 0.9}
Set grounded=false and lower confidence if you had to rely on outside knowledge or guessed."""


def generation_stage(
    query: str,
    chunks: list[dict[str, Any]],
    *,
    max_retries: int = GENERATION_MAX_RETRIES,
    base_delay: float = GENERATION_RETRY_BASE_DELAY_S,
) -> dict[str, Any]:
    """
    Generate a grounded answer using the local Ollama LLM.

    Input validation:
        - query must be non-empty
        - chunks must be a non-empty list

    Returns:
        StageResult with output = raw LLM text (including grounding JSON).
        Caller should run guardrails.parse_grounding_json() to extract clean answer.
    """
    from generation import GenerationError, generate_answer

    t0 = time.perf_counter()

    # ── Input validation ──────────────────────────────────────────────────────
    if not query or not query.strip():
        return _fail("generation", "Query is empty.", 0.0)
    if not chunks:
        return _fail("generation", "No context chunks provided — cannot generate grounded answer.", 0.0)

    # ── Generate with retry ───────────────────────────────────────────────────
    # generate_answer has its own strict grounding system prompt.
    # The grounding guard (guardrails.check_grounding) handles post-generation verification.
    result, retries, err = _retry_loop(
        generate_answer,
        (query, chunks),
        {},
        max_retries=max_retries,
        base_delay=base_delay,
        stage="generation",
    )
    elapsed = (time.perf_counter() - t0) * 1000

    if err:
        return _fail("generation", err, elapsed, retries)

    return _ok("generation", result, elapsed, retries)
