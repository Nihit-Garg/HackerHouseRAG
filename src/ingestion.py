"""
ingestion.py — Load, filter, clean ai4bharat/MSMARCO-XI passages via DuckDB.

Dataset schema (discovered via DuckDB inspection):
  source_lang, target_lang, meta, Answer, query_id, query_type,
  passages (struct: {passage_text: list[str], is_selected: list[int], url: list[str]}),
  Eng_Query, Eng_Answer, query

We UNNEST passages.passage_text so each passage becomes its own row.
Eng_Query gives the original English question for each passage set.

Why DuckDB:
  - HuggingFace `datasets` streaming fails with pyarrow nested-type error on this dataset
  - DuckDB's native parquet reader handles nested structs correctly
  - hf:// protocol reads via HTTP range requests — no file downloaded to disk
"""
from __future__ import annotations

import logging
import re
from typing import Any

import duckdb
import pandas as pd

from config import (
    DATASET_NAME,
    DATASET_SPLIT,
    LANGUAGE_FILTER,
    MSMARCO_SUBSET_SIZE,
)

logger = logging.getLogger(__name__)

_HF_PARQUET = f"hf://datasets/{DATASET_NAME}/train/asmtrain.parquet"


# ── Text cleaning ──────────────────────────────────────────────────────────────

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"\s{2,}")


def _clean_text(text: str) -> str:
    """Strip HTML tags and normalise whitespace."""
    text = _HTML_TAG_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def _is_valid(text: str) -> bool:
    """Return False for empty or very short passages (< 5 words)."""
    return bool(text) and len(text.split()) >= 5


# ── DuckDB helpers ─────────────────────────────────────────────────────────────

def _make_conn() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect()
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    return conn


# ── Main loader ────────────────────────────────────────────────────────────────

def load_passages(
    subset_size: int = MSMARCO_SUBSET_SIZE,
    language_filter: str | None = LANGUAGE_FILTER or None,
) -> list[dict[str, Any]]:
    """
    Load and clean passages from ai4bharat/MSMARCO-XI using DuckDB.

    Uses UNNEST on passages.passage_text to flatten the nested array into rows.
    Each source row has multiple passages; this yields one row per passage.

    Args:
        subset_size: Max passages to return after cleaning.
        language_filter: Filter on source_lang (e.g. 'as', 'hi').
                         None = all languages.

    Returns:
        List of dicts: {passage_id, text, query, lang}
    """
    lang_filter = language_filter.strip() if language_filter else None

    logger.info(
        "Loading %d passages from %s via DuckDB (lang=%s)…",
        subset_size, DATASET_NAME, lang_filter or "all",
    )

    conn = _make_conn()

    # Fetch 3× subset_size rows to have buffer for dedup + empty passages.
    # UNNEST expands passages.passage_text list → one row per passage.
    fetch_limit = subset_size * 3

    where_clause = f"WHERE source_lang = '{lang_filter}'" if lang_filter else ""

    sql = f"""
        SELECT
            UNNEST(passages['English_passages'])  AS passage_text,
            Eng_Query                              AS query,
            CAST(query_id AS VARCHAR)              AS query_id,
            source_lang                            AS lang,
            ROW_NUMBER() OVER ()                   AS row_num
        FROM read_parquet('{_HF_PARQUET}', union_by_name=true)
        {where_clause}
        LIMIT {fetch_limit}
    """

    logger.info("Executing DuckDB query (fetch_limit=%d)…", fetch_limit)
    try:
        df: pd.DataFrame = conn.execute(sql).df()
    except Exception as exc:
        raise RuntimeError(f"DuckDB query failed: {exc}") from exc

    logger.info("DuckDB returned %d rows (before dedup/filter).", len(df))

    # ── Process rows into passage dicts ───────────────────────────────────────
    passages: list[dict[str, Any]] = []
    seen_texts: set[str] = set()

    for _, row in df.iterrows():
        if len(passages) >= subset_size:
            break

        raw_text = row.get("passage_text")
        if not isinstance(raw_text, str):
            continue

        cleaned = _clean_text(raw_text)

        if not _is_valid(cleaned):
            continue
        if cleaned in seen_texts:
            continue
        seen_texts.add(cleaned)

        passages.append(
            {
                "passage_id": str(row.get("query_id", row.get("row_num", len(passages)))),
                "text": cleaned,
                "query": str(row["query"]) if pd.notna(row.get("query")) else None,
                "lang": str(row.get("lang", lang_filter or "unknown")),
            }
        )

    logger.info(
        "Done — collected %d clean passages from %d DuckDB rows.",
        len(passages), len(df),
    )
    return passages
