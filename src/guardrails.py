"""
guardrails.py — Four independent guardrail functions for the RAG pipeline.

Guard 1 — Unsafe Input:   keyword blocklist on raw query text (pre-STT)
Guard 2 — Off-Topic:      cosine similarity between query embedding and corpus centroid
Guard 3 — Retrieval Threshold: if best RRF score < threshold, skip LLM entirely
Guard 4 — Grounding Check: LLM outputs structured JSON with grounded/confidence fields

Each guard returns a typed GuardResult dict so callers can log uniformly.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import numpy as np

from config import (
    CORPUS_CENTROID_PATH,
    GROUNDING_CONFIDENCE_THRESHOLD,
    OFF_TOPIC_THRESHOLD,
    RETRIEVAL_SCORE_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared result type
# ─────────────────────────────────────────────────────────────────────────────

def _result(passed: bool, guard: str, reason: str, detail: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "passed": passed,
        "guard": guard,
        "reason": reason,
        "detail": detail or {},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Guard 1 — Unsafe Input
# ─────────────────────────────────────────────────────────────────────────────

# Lightweight blocklist — covers profanity, self-harm, violence, PII fishing.
# Extend freely; this is the first, cheapest gate so err on the side of coverage.
_UNSAFE_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(fuck|shit|bitch|asshole|cunt|nigger|faggot)\b",       # profanity
        r"\b(kill\s+(your|my|him|her|them|people))\b",              # violence intent
        r"\b(how\s+to\s+(make|build|create)\s+(?:\w+\s+)?(bomb|weapon|poison|malware|exploit))\b",
        r"\b(suicide|self.?harm|cut\s+myself|hang\s+myself)\b",     # self-harm
        r"\b(social\s+security\s+number|credit\s+card\s+number|cvv|ssn)\b",  # PII fishing
        r"\b(jailbreak|ignore\s+(previous|all)\s+instructions?|prompt\s+injection)\b",
    ]
]


def check_unsafe_input(query: str) -> dict[str, Any]:
    """
    Guard 1: Scan raw query text against an unsafe keyword/pattern blocklist.

    Args:
        query: Raw text query (before or after STT).

    Returns:
        GuardResult — passed=True means query is safe to proceed.
    """
    if not query or not query.strip():
        return _result(False, "unsafe_input", "Empty query", {"matched_pattern": None})

    for pattern in _UNSAFE_PATTERNS:
        match = pattern.search(query)
        if match:
            logger.warning("Unsafe input detected | pattern=%s | query=%r", pattern.pattern, query[:80])
            return _result(
                False, "unsafe_input",
                "Query contains unsafe or prohibited content.",
                {"matched_pattern": pattern.pattern, "matched_text": match.group(0)},
            )

    return _result(True, "unsafe_input", "Query passed safety check.")


# ─────────────────────────────────────────────────────────────────────────────
# Guard 2 — Off-Topic Detection
# ─────────────────────────────────────────────────────────────────────────────

_centroid_cache: np.ndarray | None = None


def _load_centroid() -> np.ndarray | None:
    global _centroid_cache
    if _centroid_cache is not None:
        return _centroid_cache
    if not CORPUS_CENTROID_PATH.exists():
        logger.warning("Corpus centroid not found at %s — off-topic guard disabled.", CORPUS_CENTROID_PATH)
        return None
    _centroid_cache = np.load(str(CORPUS_CENTROID_PATH)).astype("float32")
    return _centroid_cache


def check_off_topic(query_embedding: np.ndarray, threshold: float = OFF_TOPIC_THRESHOLD) -> dict[str, Any]:
    """
    Guard 2: Compare query embedding to the corpus centroid via cosine similarity.

    If similarity < threshold, the query is likely off-topic for this corpus.

    Args:
        query_embedding: 1-D float32 array (already L2-normalised by embeddings.py).
        threshold: Minimum cosine similarity to be considered on-topic.

    Returns:
        GuardResult — passed=True means query is on-topic.
    """
    centroid = _load_centroid()
    if centroid is None:
        # Guard disabled — allow through
        return _result(True, "off_topic", "Off-topic guard disabled (centroid not found).")

    q = query_embedding.astype("float32").flatten()
    c = centroid.flatten()

    # Embeddings from BGE are L2-normalised; dot product = cosine similarity
    norm_q = np.linalg.norm(q)
    norm_c = np.linalg.norm(c)
    if norm_q == 0 or norm_c == 0:
        return _result(True, "off_topic", "Could not compute similarity (zero vector).")

    similarity = float(np.dot(q / norm_q, c / norm_c))

    if similarity < threshold:
        logger.info("Off-topic query | similarity=%.4f < threshold=%.4f", similarity, threshold)
        return _result(
            False, "off_topic",
            f"Query appears off-topic for this corpus (similarity={similarity:.3f} < {threshold}).",
            {"cosine_similarity": similarity, "threshold": threshold},
        )

    return _result(
        True, "off_topic",
        f"Query is on-topic (similarity={similarity:.3f}).",
        {"cosine_similarity": similarity, "threshold": threshold},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Guard 3 — Retrieval Threshold (Refusal before LLM call)
# ─────────────────────────────────────────────────────────────────────────────

def check_retrieval_threshold(
    chunks: list[dict[str, Any]],
    threshold: float = RETRIEVAL_SCORE_THRESHOLD,
) -> dict[str, Any]:
    """
    Guard 3: Check if the best-ranked retrieved chunk has a meaningful RRF score.

    If max(rrf_score) < threshold, the retriever couldn't find relevant context.
    Skip the LLM call entirely and return "insufficient context" — no hallucination risk.

    Args:
        chunks: List of retrieved chunk dicts (each has 'rrf_score').
        threshold: Minimum RRF score for the top result.

    Returns:
        GuardResult — passed=True means retrieval found sufficient context.
    """
    if not chunks:
        return _result(
            False, "retrieval_threshold",
            "No chunks retrieved — insufficient context to answer.",
            {"max_rrf_score": 0.0, "threshold": threshold},
        )

    max_rrf = max(c.get("rrf_score", 0.0) for c in chunks)

    if max_rrf < threshold:
        logger.info("Low retrieval score | max_rrf=%.4f < threshold=%.4f", max_rrf, threshold)
        return _result(
            False, "retrieval_threshold",
            f"Retrieved context is too weak to answer reliably (max_rrf={max_rrf:.4f} < {threshold}).",
            {"max_rrf_score": max_rrf, "threshold": threshold},
        )

    return _result(
        True, "retrieval_threshold",
        f"Retrieval threshold passed (max_rrf={max_rrf:.4f}).",
        {"max_rrf_score": max_rrf, "threshold": threshold},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Guard 4 — Grounding Check (post-generation)
# ─────────────────────────────────────────────────────────────────────────────

# Matches the JSON block we ask the LLM to append to its answer
_GROUNDING_JSON_RE = re.compile(
    r"\{[^{}]*\"grounded\"\s*:\s*(true|false)[^{}]*\"confidence\"\s*:\s*([0-9.]+)[^{}]*\}",
    re.IGNORECASE | re.DOTALL,
)


def parse_grounding_json(raw_answer: str) -> tuple[str, bool, float]:
    """
    Extract structured grounding metadata appended by the LLM to its answer.

    The generation prompt asks the LLM to append:
        {"grounded": true/false, "confidence": 0.0-1.0}
    at the end of its response.

    Returns:
        (clean_answer, grounded, confidence)
    """
    match = _GROUNDING_JSON_RE.search(raw_answer)
    if match:
        grounded_str = match.group(1).lower() == "true"
        try:
            confidence = float(match.group(2))
        except ValueError:
            confidence = 1.0 if grounded_str else 0.0
        clean_answer = raw_answer[: match.start()].strip()
        return clean_answer, grounded_str, min(max(confidence, 0.0), 1.0)

    # No JSON found — assume grounded (LLM may have ignored instruction)
    return raw_answer.strip(), True, 1.0


def check_grounding(
    answer: str,
    chunks: list[dict[str, Any]],
    threshold: float = GROUNDING_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """
    Guard 4: Verify the generated answer is grounded in retrieved context.

    Two signals:
      1. LLM self-report: parses {"grounded": bool, "confidence": float} from answer.
      2. Term overlap: fraction of answer bigrams appearing in retrieved chunks.

    Args:
        answer: Raw LLM output (may contain appended JSON grounding block).
        chunks: Retrieved chunk dicts (each has 'text').
        threshold: Minimum confidence to accept the answer.

    Returns:
        GuardResult with 'clean_answer' in detail (JSON stripped).
    """
    clean_answer, grounded, confidence = parse_grounding_json(answer)

    # Term overlap as a secondary signal
    answer_tokens = set(re.findall(r"\b\w{4,}\b", clean_answer.lower()))
    corpus_tokens = set()
    for chunk in chunks:
        corpus_tokens.update(re.findall(r"\b\w{4,}\b", chunk.get("text", "").lower()))

    if answer_tokens:
        overlap = len(answer_tokens & corpus_tokens) / len(answer_tokens)
    else:
        overlap = 0.0

    detail = {
        "clean_answer": clean_answer,
        "llm_grounded": grounded,
        "llm_confidence": confidence,
        "term_overlap": round(overlap, 3),
        "threshold": threshold,
    }

    # Fail if LLM explicitly says ungrounded OR confidence is below threshold
    if not grounded or confidence < threshold:
        logger.warning(
            "Grounding check failed | grounded=%s confidence=%.2f overlap=%.2f",
            grounded, confidence, overlap,
        )
        return _result(
            False, "grounding",
            f"Answer not sufficiently grounded in retrieved context "
            f"(llm_grounded={grounded}, confidence={confidence:.2f}).",
            detail,
        )

    return _result(True, "grounding", f"Answer is grounded (confidence={confidence:.2f}).", detail)
