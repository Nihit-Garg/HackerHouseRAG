"""
config.py — Single source of truth for all constants and environment variables.
Load order: defaults → .env file → actual environment variables (env wins).
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ── Load .env from repo root ────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env", override=False)  # env vars already set take priority

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR: Path = _REPO_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
INDEX_DIR: Path = DATA_DIR / "index"

FAISS_INDEX_PATH: Path = INDEX_DIR / "faiss.index"
FAISS_META_PATH: Path = INDEX_DIR / "faiss_meta.json"
BM25_INDEX_PATH: Path = INDEX_DIR / "bm25.pkl"
CHUNKS_PATH: Path = PROCESSED_DIR / "chunks.json"
CORPUS_CENTROID_PATH: Path = INDEX_DIR / "corpus_centroid.npy"  # mean of all corpus embeddings

# ── Data ─────────────────────────────────────────────────────────────────────
# ai4bharat/MSMARCO-XI — multilingual MSMARCO dataset.
# Use streaming=True in ingestion to avoid downloading all 20+ language files.
DATASET_NAME: str = "ai4bharat/MSMARCO-XI"
DATASET_SPLIT: str = "train"
MSMARCO_SUBSET_SIZE: int = int(os.getenv("MSMARCO_SUBSET_SIZE", "5000"))
# Set to "en" to attempt English filtering (slower), or "" to take all languages (fastest).
LANGUAGE_FILTER: str = os.getenv("LANGUAGE_FILTER", "")

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE_TOKENS: int = int(os.getenv("CHUNK_SIZE_TOKENS", "256"))
CHUNK_OVERLAP_PCT: float = float(os.getenv("CHUNK_OVERLAP_PCT", "0.2"))  # 20%
CHUNKING_STRATEGY: str = os.getenv("CHUNKING_STRATEGY", "metadata")  # fixed | sentence | metadata

# ── Embedding model ───────────────────────────────────────────────────────────
# BAAI/bge-small-en-v1.5 chosen for speed + quality balance at this data scale.
# ~33M params, 384-dim embeddings, strong on retrieval benchmarks.
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

# ── FAISS ─────────────────────────────────────────────────────────────────────
# Flat index (exact search) is fine at ≤10k passages. Swap to IVF for scale.
FAISS_INDEX_TYPE: str = os.getenv("FAISS_INDEX_TYPE", "flat")  # flat | ivf
FAISS_NLIST: int = int(os.getenv("FAISS_NLIST", "100"))  # only used for IVF

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K: int = int(os.getenv("TOP_K", "5"))
RRF_K_CONSTANT: int = int(os.getenv("RRF_K_CONSTANT", "60"))  # standard RRF k param

# ── LLM ───────────────────────────────────────────────────────────────────────
# qwen2.5:7b — strong instruction following, fits 8GB VRAM, good for RAG grounding.
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT_S: int = int(os.getenv("OLLAMA_TIMEOUT_S", "120"))

# ── Sarvam STT ────────────────────────────────────────────────────────────────
SARVAM_API_KEY: str | None = os.getenv("SARVAM_API_KEY")
# TODO: verify exact endpoint path against current Sarvam docs
SARVAM_STT_URL: str = os.getenv(
    "SARVAM_STT_URL",
    "https://api.sarvam.ai/speech-to-text",
)
SARVAM_LANGUAGE_CODE: str = os.getenv("SARVAM_LANGUAGE_CODE", "en-IN")
SARVAM_MODEL: str = os.getenv("SARVAM_MODEL", "saarika:v2.5")
STT_MAX_RETRIES: int = int(os.getenv("STT_MAX_RETRIES", "2"))
STT_RETRY_BASE_DELAY_S: float = float(os.getenv("STT_RETRY_BASE_DELAY_S", "1.0"))

# ── FastAPI ────────────────────────────────────────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
API_RELOAD: bool = os.getenv("API_RELOAD", "true").lower() == "true"

# ── Day 2: Guardrails ──────────────────────────────────────────────────────────
# Guard 1: Unsafe input — keyword blocklist (inline in guardrails.py)
# Guard 2: Off-topic — cosine sim between query embedding and corpus centroid
OFF_TOPIC_THRESHOLD: float = float(os.getenv("OFF_TOPIC_THRESHOLD", "0.15"))
# Guard 3: Retrieval threshold — skip LLM if best RRF score is too low
RETRIEVAL_SCORE_THRESHOLD: float = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.02"))
# Guard 4: Grounding — minimum confidence for LLM self-grounding score
GROUNDING_CONFIDENCE_THRESHOLD: float = float(os.getenv("GROUNDING_CONFIDENCE_THRESHOLD", "0.5"))

# ── Day 2: Harness retries ─────────────────────────────────────────────────────
GENERATION_MAX_RETRIES: int = int(os.getenv("GENERATION_MAX_RETRIES", "2"))
GENERATION_RETRY_BASE_DELAY_S: float = float(os.getenv("GENERATION_RETRY_BASE_DELAY_S", "1.0"))

# ── Day 2: Benchmark ──────────────────────────────────────────────────────────
BENCHMARK_N_QUERIES: int = int(os.getenv("BENCHMARK_N_QUERIES", "50"))
