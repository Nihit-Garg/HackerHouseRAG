"""
api.py — FastAPI server exposing the RAG pipeline over HTTP.

Endpoints:
  POST /query/text        → text query directly
  POST /query/audio       → multipart audio file upload → STT → RAG
  GET  /health            → liveness check
  GET  /index/status      → check if FAISS/BM25 indexes exist

Run with:
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import (
    BM25_INDEX_PATH,
    FAISS_INDEX_PATH,
    TOP_K,
)

logger = logging.getLogger(__name__)

# ── App state (lazy-loaded on first request or at startup) ────────────────────
_pipeline: Any = None  # RAGPipeline instance


def _load_pipeline() -> Any:
    """Lazy-load the RAGPipeline (loads indexes from disk)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    from embeddings import get_embedding_model
    from graph import RAGPipeline
    from retrieval import HybridRetriever
    from vectorstore import load_bm25_index, load_faiss_index

    try:
        faiss_index, chunks = load_faiss_index()
        bm25_index = load_bm25_index()
        emb_model = get_embedding_model()
        retriever = HybridRetriever(faiss_index, bm25_index, chunks, emb_model)
        _pipeline = RAGPipeline(retriever)
        logger.info("RAGPipeline loaded successfully.")
    except FileNotFoundError as exc:
        logger.error("Index files not found: %s", exc)
        raise

    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load pipeline at startup if indexes exist."""
    if FAISS_INDEX_PATH.exists() and BM25_INDEX_PATH.exists():
        try:
            _load_pipeline()
        except Exception as exc:
            logger.warning("Could not pre-load pipeline at startup: %s", exc)
    yield


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="HackerHouse RAG API",
    description="Voice-enabled RAG pipeline: STT → Hybrid Retrieval → Grounded LLM Answer",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────

class TextQueryRequest(BaseModel):
    query: str
    top_k: int = TOP_K


class PipelineResponse(BaseModel):
    query: str
    transcribed_text: str
    retrieved_chunks: list[dict]
    answer: str
    stage_timings: dict[str, float]
    errors: list[str]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_pipeline() -> Any:
    try:
        return _load_pipeline()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Index not built. Run `python scripts/build_index.py` first.",
        )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


@app.get("/index/status", tags=["System"])
async def index_status() -> dict[str, Any]:
    """Check whether FAISS and BM25 index files exist on disk."""
    return {
        "faiss_index_exists": FAISS_INDEX_PATH.exists(),
        "bm25_index_exists": BM25_INDEX_PATH.exists(),
        "pipeline_loaded": _pipeline is not None,
    }


@app.post("/query/text", response_model=PipelineResponse, tags=["Query"])
async def query_text(request: TextQueryRequest) -> PipelineResponse:
    """
    Submit a text query directly (bypasses STT).

    Args:
        request: {query: str, top_k: int}

    Returns:
        Full pipeline response dict.
    """
    pipeline = _get_pipeline()

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    result = pipeline.answer(text_query=request.query)
    return PipelineResponse(**result)


@app.post("/query/audio", response_model=PipelineResponse, tags=["Query"])
async def query_audio(file: UploadFile = File(...)) -> PipelineResponse:
    """
    Submit an audio file for transcription and RAG answering.

    Accepts: wav, mp3, ogg, flac, m4a, webm
    """
    pipeline = _get_pipeline()

    allowed = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{suffix}'. Allowed: {sorted(allowed)}",
        )

    # Write upload to a temp file
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = pipeline.answer(audio_path=tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return PipelineResponse(**result)


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT, API_RELOAD

    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=API_RELOAD)
