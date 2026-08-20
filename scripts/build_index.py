"""
build_index.py — One-time index build script.

Steps:
  1. Load + clean MSMARCO-XI passages (ingestion)
  2. Chunk passages using configured strategy
  3. Embed chunks (sentence-transformers)
  4. Build FAISS flat index + BM25 index
  5. Save all artifacts to data/index/

Usage:
  python scripts/build_index.py
  python scripts/build_index.py --strategy sentence --chunk-size 128
  python scripts/build_index.py --force          # rebuild even if index exists

Idempotent: skips rebuild if index files already exist (use --force to override).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from config import (
    BM25_INDEX_PATH,
    CHUNK_OVERLAP_PCT,
    CHUNK_SIZE_TOKENS,
    CHUNKING_STRATEGY,
    CHUNKS_PATH,
    CORPUS_CENTROID_PATH,
    FAISS_INDEX_PATH,
    FAISS_META_PATH,
    INDEX_DIR,
    MSMARCO_SUBSET_SIZE,
    PROCESSED_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_index")


def _index_exists() -> bool:
    return FAISS_INDEX_PATH.exists() and BM25_INDEX_PATH.exists() and FAISS_META_PATH.exists()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS + BM25 indexes for the RAG pipeline.")
    parser.add_argument(
        "--strategy",
        choices=["fixed", "sentence", "metadata"],
        default=CHUNKING_STRATEGY,
        help=f"Chunking strategy (default: {CHUNKING_STRATEGY})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE_TOKENS,
        help=f"Chunk size in tokens (default: {CHUNK_SIZE_TOKENS})",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=CHUNK_OVERLAP_PCT,
        help=f"Overlap fraction for fixed/metadata strategies (default: {CHUNK_OVERLAP_PCT})",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=MSMARCO_SUBSET_SIZE,
        help=f"Number of passages to load (default: {MSMARCO_SUBSET_SIZE})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if index files already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Idempotency check ─────────────────────────────────────────────────────
    if _index_exists() and not args.force:
        logger.info(
            "Index files already exist at %s — skipping rebuild. Use --force to override.",
            INDEX_DIR,
        )
        return

    if args.force and _index_exists():
        logger.info("--force specified; rebuilding existing index.")

    # ── Step 1: Load passages ─────────────────────────────────────────────────
    logger.info("Step 1/4: Loading passages (subset=%d)…", args.subset)
    from ingestion import load_passages
    passages = load_passages(subset_size=args.subset)
    logger.info("  → %d passages loaded.", len(passages))

    # ── Step 2: Chunk ─────────────────────────────────────────────────────────
    logger.info(
        "Step 2/4: Chunking (strategy=%s, chunk_size=%d, overlap=%.0f%%)…",
        args.strategy,
        args.chunk_size,
        args.overlap * 100,
    )
    from chunking import chunk_passages
    chunks = chunk_passages(
        passages,
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        overlap_pct=args.overlap,
    )
    logger.info("  → %d chunks produced.", len(chunks))

    # Save chunks to disk (useful for inspection / reuse)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    logger.info("  → Chunks saved to %s", CHUNKS_PATH)

    # ── Step 3 & 4: Embed + build indexes ────────────────────────────────────
    logger.info("Step 3/4: Loading embedding model and embedding chunks…")
    from embeddings import get_embedding_model
    emb_model = get_embedding_model()

    logger.info("Step 4/4: Building FAISS + BM25 indexes…")
    from vectorstore import build_all_indexes
    faiss_index, bm25_index = build_all_indexes(chunks, emb_model)

    logger.info("✓ Index build complete.")
    logger.info("  FAISS index: %d vectors at %s", faiss_index.ntotal, FAISS_INDEX_PATH)
    logger.info("  BM25 index: %s", BM25_INDEX_PATH)
    logger.info("  Chunk count: %d at %s", len(chunks), CHUNKS_PATH)

    # ── Step 5: Compute corpus centroid (for off-topic guard) ──────────────────
    logger.info("Step 5/5: Computing corpus centroid for off-topic guardrail…")
    try:
        import numpy as np
        # Reconstruct all vectors from FAISS index
        n = faiss_index.ntotal
        all_vectors = np.zeros((n, faiss_index.d), dtype="float32")
        faiss_index.reconstruct_n(0, n, all_vectors)
        centroid = all_vectors.mean(axis=0)
        np.save(str(CORPUS_CENTROID_PATH), centroid)
        logger.info("  Corpus centroid saved → %s (dim=%d)", CORPUS_CENTROID_PATH, len(centroid))
    except Exception as exc:
        logger.warning("  Could not compute centroid: %s — off-topic guard will be disabled.", exc)


if __name__ == "__main__":
    main()
