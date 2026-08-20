"""
chunking.py — Three chunking strategies for RAG preprocessing.

All strategies return List[Chunk] where each Chunk is a TypedDict:
    {text: str, metadata: ChunkMetadata}

ChunkMetadata always contains:
    passage_id, chunk_id, source_query, position_in_doc, strategy_used
"""
from __future__ import annotations

import logging
import re
from typing import Any, TypedDict

import tiktoken

from config import (
    CHUNK_OVERLAP_PCT,
    CHUNK_SIZE_TOKENS,
)

logger = logging.getLogger(__name__)

# ── Shared tokenizer (cl100k_base ≈ GPT-4 / most modern LLMs) ─────────────────
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


# ── Types ──────────────────────────────────────────────────────────────────────

class ChunkMetadata(TypedDict):
    passage_id: str
    chunk_id: str
    source_query: str | None
    position_in_doc: int        # 0-indexed chunk position within the passage
    strategy_used: str


class Chunk(TypedDict):
    text: str
    metadata: ChunkMetadata


# ── Tokenizer helpers ──────────────────────────────────────────────────────────

def _encode(text: str) -> list[int]:
    return _TOKENIZER.encode(text, disallowed_special=())


def _decode(tokens: list[int]) -> str:
    return _TOKENIZER.decode(tokens)


def _token_len(text: str) -> int:
    return len(_encode(text))


# ── Strategy 1: Fixed-size with overlap ───────────────────────────────────────

def chunk_fixed(
    passage: dict[str, Any],
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap_pct: float = CHUNK_OVERLAP_PCT,
) -> list[Chunk]:
    """
    Token-based fixed-size chunking with configurable overlap.

    Args:
        passage: Dict with keys: passage_id, text, query.
        chunk_size: Target chunk size in tokens.
        overlap_pct: Fraction of chunk_size to overlap (e.g. 0.2 = 20%).

    Returns:
        List of Chunk dicts.
    """
    overlap = int(chunk_size * overlap_pct)
    tokens = _encode(passage["text"])
    chunks: list[Chunk] = []

    start = 0
    position = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = _decode(chunk_tokens).strip()

        if chunk_text:
            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata=ChunkMetadata(
                        passage_id=passage["passage_id"],
                        chunk_id=f"{passage['passage_id']}_fixed_{position}",
                        source_query=passage.get("query"),
                        position_in_doc=position,
                        strategy_used="fixed",
                    ),
                )
            )
            position += 1

        if end == len(tokens):
            break
        start = end - overlap  # step back by overlap

    return chunks


# ── Strategy 2: Sentence / semantic chunking ──────────────────────────────────

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences.
    Tries NLTK first; falls back to regex on import error.
    Note: For true semantic breakpoint detection, sentence embeddings can be
    used to find cosine-similarity drops between adjacent sentences — extend here.
    """
    try:
        import nltk  # noqa: PLC0415
        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
            sentences = nltk.sent_tokenize(text)
        return sentences
    except ImportError:
        logger.warning("NLTK not available; falling back to regex sentence splitter.")
        return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def chunk_sentence(
    passage: dict[str, Any],
    chunk_size: int = CHUNK_SIZE_TOKENS,
) -> list[Chunk]:
    """
    Greedily merge sentences up to the token budget.

    Args:
        passage: Dict with keys: passage_id, text, query.
        chunk_size: Max tokens per chunk.

    Returns:
        List of Chunk dicts.
    """
    sentences = _split_sentences(passage["text"])
    chunks: list[Chunk] = []
    current_sentences: list[str] = []
    current_tokens = 0
    position = 0

    for sentence in sentences:
        sent_tokens = _token_len(sentence)

        if current_tokens + sent_tokens > chunk_size and current_sentences:
            # Flush current buffer
            chunk_text = " ".join(current_sentences).strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        metadata=ChunkMetadata(
                            passage_id=passage["passage_id"],
                            chunk_id=f"{passage['passage_id']}_sentence_{position}",
                            source_query=passage.get("query"),
                            position_in_doc=position,
                            strategy_used="sentence",
                        ),
                    )
                )
                position += 1
            current_sentences = [sentence]
            current_tokens = sent_tokens
        else:
            current_sentences.append(sentence)
            current_tokens += sent_tokens

    # Flush remaining
    if current_sentences:
        chunk_text = " ".join(current_sentences).strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata=ChunkMetadata(
                        passage_id=passage["passage_id"],
                        chunk_id=f"{passage['passage_id']}_sentence_{position}",
                        source_query=passage.get("query"),
                        position_in_doc=position,
                        strategy_used="sentence",
                    ),
                )
            )

    return chunks


# ── Strategy 3: Metadata-aware chunking ──────────────────────────────────────

def chunk_metadata_aware(
    passage: dict[str, Any],
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap_pct: float = CHUNK_OVERLAP_PCT,
) -> list[Chunk]:
    """
    Fixed-size chunking (with overlap) that enriches each chunk with full
    structured metadata for citation and filtering use cases.

    The metadata schema is identical to the other strategies — this strategy
    ensures every field is populated even when the source passage is sparse.

    Args:
        passage: Dict with keys: passage_id, text, query.
        chunk_size: Target chunk size in tokens.
        overlap_pct: Fraction of chunk_size to overlap.

    Returns:
        List of Chunk dicts.
    """
    # Reuse fixed chunking logic but override strategy_used
    raw_chunks = chunk_fixed(passage, chunk_size=chunk_size, overlap_pct=overlap_pct)

    enriched: list[Chunk] = []
    total = len(raw_chunks)
    for chunk in raw_chunks:
        meta = chunk["metadata"]
        position = meta["position_in_doc"]
        enriched.append(
            Chunk(
                text=chunk["text"],
                metadata=ChunkMetadata(
                    passage_id=meta["passage_id"],
                    chunk_id=meta["chunk_id"].replace("_fixed_", "_meta_"),
                    source_query=meta["source_query"] or "",
                    position_in_doc=position,
                    strategy_used="metadata",
                ),
            )
        )
    return enriched


# ── Public dispatcher ─────────────────────────────────────────────────────────

def chunk_passage(
    passage: dict[str, Any],
    strategy: str = "metadata",
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap_pct: float = CHUNK_OVERLAP_PCT,
) -> list[Chunk]:
    """
    Dispatch to the requested chunking strategy.

    Args:
        passage: Passage dict from ingestion.
        strategy: One of 'fixed', 'sentence', 'metadata'.
        chunk_size: Token budget per chunk.
        overlap_pct: Overlap fraction (used by fixed/metadata strategies).

    Returns:
        List of Chunk dicts.

    Raises:
        ValueError: If strategy is not recognised.
    """
    if strategy == "fixed":
        return chunk_fixed(passage, chunk_size=chunk_size, overlap_pct=overlap_pct)
    elif strategy == "sentence":
        return chunk_sentence(passage, chunk_size=chunk_size)
    elif strategy == "metadata":
        return chunk_metadata_aware(passage, chunk_size=chunk_size, overlap_pct=overlap_pct)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy!r}. Choose from: fixed, sentence, metadata.")


def chunk_passages(
    passages: list[dict[str, Any]],
    strategy: str = "metadata",
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap_pct: float = CHUNK_OVERLAP_PCT,
) -> list[Chunk]:
    """Chunk a list of passages using the specified strategy."""
    all_chunks: list[Chunk] = []
    for passage in passages:
        try:
            all_chunks.extend(
                chunk_passage(passage, strategy=strategy, chunk_size=chunk_size, overlap_pct=overlap_pct)
            )
        except Exception as exc:
            logger.warning("Failed to chunk passage %s: %s", passage.get("passage_id"), exc)
    logger.info("Chunked %d passages → %d chunks (strategy=%s)", len(passages), len(all_chunks), strategy)
    return all_chunks
