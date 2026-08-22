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
import shutil
import subprocess
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
        # Day 2: pass embedder so off-topic guard can embed queries
        _pipeline = RAGPipeline(retriever, emb_model)
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
    guardrail_triggered: str | None = None
    guardrail_detail: dict[str, Any] = {}


# ── Audio helpers ─────────────────────────────────────────────────────────────

# Formats Sarvam reliably accepts as wav-encoded input
_NEEDS_CONVERSION = {".webm", ".ogg"}


def _convert_to_wav(src: Path) -> Path | None:
    """
    Convert audio to 16kHz mono WAV using ffmpeg.
    Returns a new temp Path on success, or None if ffmpeg is unavailable.
    The caller is responsible for deleting the returned file.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    out_path = Path(tmp.name)

    cmd = [
        ffmpeg,
        "-y",           # overwrite without asking
        "-i", str(src), # input file
        "-ar", "16000", # 16kHz — Sarvam STT preferred sample rate
        "-ac", "1",     # mono
        "-f", "wav",    # force WAV output
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        logger.info("Converted %s → %s (wav/16kHz/mono)", src.name, out_path.name)
        return out_path
    except subprocess.CalledProcessError as exc:
        logger.error("ffmpeg conversion failed: %s", exc.stderr.decode(errors="ignore"))
        out_path.unlink(missing_ok=True)
        return None
    except Exception as exc:
        logger.error("ffmpeg error: %s", exc)
        out_path.unlink(missing_ok=True)
        return None


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
    faiss_exists = FAISS_INDEX_PATH.exists()

    total_chunks = None
    if faiss_exists:
        try:
            import faiss

            total_chunks = faiss.read_index(str(FAISS_INDEX_PATH)).ntotal
        except Exception as exc:
            logger.warning("Could not read FAISS index for status check: %s", exc)

    return {
        "faiss_index_exists": faiss_exists,
        "bm25_index_exists": BM25_INDEX_PATH.exists(),
        "pipeline_loaded": _pipeline is not None,
        "total_chunks": total_chunks,
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
    webm/ogg are automatically converted to 16kHz mono WAV via ffmpeg.
    """
    pipeline = _get_pipeline()

    allowed = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"}
    suffix = Path(file.filename or "audio.webm").suffix.lower() or ".webm"
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{suffix}'. Allowed: {sorted(allowed)}",
        )

    # Write upload to a temp file
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    converted_path: Path | None = None
    try:
        audio_path = tmp_path

        # Convert webm/ogg → wav so Sarvam STT gets a clean PCM wav
        if suffix in _NEEDS_CONVERSION:
            converted_path = _convert_to_wav(tmp_path)
            if converted_path:
                audio_path = converted_path
            else:
                logger.warning(
                    "ffmpeg not found — sending %s directly to Sarvam (may fail).", suffix
                )

        result = pipeline.answer(audio_path=str(audio_path))
    finally:
        tmp_path.unlink(missing_ok=True)
        if converted_path:
            converted_path.unlink(missing_ok=True)

    return PipelineResponse(**result)


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT, API_RELOAD

    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=API_RELOAD)
