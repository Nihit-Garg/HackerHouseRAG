"""
generation.py — Ollama LLM call with context-grounded RAG prompt.

Model: qwen2.5:7b (configurable via OLLAMA_MODEL env var)
Client: ollama Python client → REST fallback if client unavailable

The system prompt strictly restricts the model to the provided context.
If the answer is not found, the model must return FALLBACK_RESPONSE exactly.
"""
from __future__ import annotations

import logging
from typing import Any

from config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT_S

logger = logging.getLogger(__name__)

# Exact fallback string the model is instructed to return when context is insufficient
FALLBACK_RESPONSE = "I don't have enough information in the provided context to answer that."

_SYSTEM_PROMPT = f"""\
You are a precise question-answering assistant.

STRICT RULES — follow them exactly:
1. Answer ONLY using the numbered context blocks provided below.
2. Do NOT use any outside knowledge, training data, or speculation.
3. If the answer cannot be found in the context, respond with EXACTLY this sentence and nothing else:
   "{FALLBACK_RESPONSE}"
4. When you use information from a context block, reference it as [Block N].
5. Be concise and factual. Do not pad your answer.
"""


def _format_context(retrieved_chunks: list[dict[str, Any]]) -> str:
    """
    Format retrieved chunks as numbered context blocks for the prompt.

    Args:
        retrieved_chunks: List of retrieval result dicts (must have 'text' key).

    Returns:
        Formatted context string.
    """
    lines = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        lines.append(f"[Block {i}]\n{chunk['text'].strip()}")
    return "\n\n".join(lines)


def _build_user_message(query: str, context: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {query}"


# ── Ollama client call ────────────────────────────────────────────────────────

def _generate_via_ollama_client(
    query: str,
    context: str,
    model: str,
) -> str:
    """Use the official `ollama` Python package."""
    import ollama  # noqa: PLC0415

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_message(query, context)},
    ]
    response = ollama.chat(
        model=model,
        messages=messages,
        options={"temperature": 0.0},  # deterministic for grounded RAG
    )
    return response["message"]["content"].strip()


def _generate_via_rest(
    query: str,
    context: str,
    model: str,
    base_url: str,
    timeout: int,
) -> str:
    """Fallback: direct REST call to Ollama /api/chat."""
    import requests  # noqa: PLC0415

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_message(query, context)},
        ],
        "stream": False,
        "options": {"temperature": 0.0},
    }
    response = requests.post(
        f"{base_url}/api/chat",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


# ── Public API ────────────────────────────────────────────────────────────────

class GenerationError(Exception):
    """Raised when LLM generation fails."""


def generate_answer(
    query: str,
    retrieved_chunks: list[dict[str, Any]],
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    timeout: int = OLLAMA_TIMEOUT_S,
) -> str:
    """
    Generate a grounded answer from the query and retrieved chunks.

    Args:
        query: User query string.
        retrieved_chunks: List of {text, metadata, …} dicts from retrieval.
        model: Ollama model name.
        base_url: Ollama server URL.
        timeout: Request timeout in seconds.

    Returns:
        Generated answer string. Returns FALLBACK_RESPONSE if context is insufficient.

    Raises:
        GenerationError: If the LLM call fails.
    """
    if not retrieved_chunks:
        logger.warning("No chunks provided to generation — returning fallback.")
        return FALLBACK_RESPONSE

    context = _format_context(retrieved_chunks)
    logger.info("Generating answer with model %s (%d context blocks)…", model, len(retrieved_chunks))

    try:
        answer = _generate_via_ollama_client(query, context, model)
        logger.info("Answer generated via ollama client.")
        return answer

    except ImportError:
        logger.warning("`ollama` package not installed; falling back to REST API.")

    except Exception as exc:
        logger.warning("ollama client failed (%s); falling back to REST API.", exc)

    try:
        answer = _generate_via_rest(query, context, model, base_url, timeout)
        logger.info("Answer generated via REST fallback.")
        return answer

    except Exception as exc:
        raise GenerationError(f"Ollama generation failed: {exc}") from exc
