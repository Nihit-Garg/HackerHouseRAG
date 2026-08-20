"""
graph.py — LangGraph state graph for the RAG pipeline.

Graph nodes:
  stt_node       → transcribe audio (or pass text directly)
  retrieval_node → hybrid dense+BM25 retrieval
  generation_node → Ollama grounded answer

State flows: START → stt_node → retrieval_node → generation_node → END
Errors in any node are captured in state['errors'] and short-circuit safely.
"""
from __future__ import annotations

import logging
import time
from typing import Annotated, Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from config import TOP_K
from generation import FALLBACK_RESPONSE, GenerationError, generate_answer
from retrieval import HybridRetriever
from stt import STTError, transcribe

logger = logging.getLogger(__name__)


# ── State schema ──────────────────────────────────────────────────────────────

class PipelineState(TypedDict, total=False):
    # Inputs
    audio_path: str | None
    text_query: str | None         # direct text input (bypass STT)

    # Intermediate + outputs
    transcribed_text: str
    retrieved_chunks: list[dict[str, Any]]
    answer: str

    # Timing (ms)
    stt_start: float
    stt_end: float
    retrieval_start: float
    retrieval_end: float
    generation_start: float
    generation_end: float

    # Error list — accumulated, never thrown
    errors: list[str]


# ── Node implementations ───────────────────────────────────────────────────────

def stt_node(state: PipelineState) -> PipelineState:
    """Transcribe audio → text, or pass text_query through."""
    updates: dict[str, Any] = {"stt_start": time.time(), "errors": state.get("errors", [])}

    # Short-circuit: text query provided directly
    if state.get("text_query"):
        updates["transcribed_text"] = state["text_query"]
        updates["stt_end"] = time.time()
        logger.info("STT skipped — using direct text query.")
        return {**state, **updates}  # type: ignore[return-value]

    audio_path = state.get("audio_path")
    if not audio_path:
        updates["errors"] = updates["errors"] + ["STT: no audio_path or text_query provided."]
        updates["transcribed_text"] = ""
        updates["stt_end"] = time.time()
        return {**state, **updates}  # type: ignore[return-value]

    try:
        transcript = transcribe(audio_path)
        updates["transcribed_text"] = transcript
    except (STTError, FileNotFoundError, ValueError) as exc:
        error_msg = f"STT failed: {exc}"
        logger.error(error_msg)
        updates["errors"] = updates["errors"] + [error_msg]
        updates["transcribed_text"] = ""
    except Exception as exc:
        error_msg = f"STT unexpected error: {exc}"
        logger.exception(error_msg)
        updates["errors"] = updates["errors"] + [error_msg]
        updates["transcribed_text"] = ""

    updates["stt_end"] = time.time()
    return {**state, **updates}  # type: ignore[return-value]


def make_retrieval_node(retriever: HybridRetriever, top_k: int = TOP_K):
    """Factory — closes over the retriever instance."""

    def retrieval_node(state: PipelineState) -> PipelineState:
        updates: dict[str, Any] = {
            "retrieval_start": time.time(),
            "errors": state.get("errors", []),
        }

        query = state.get("transcribed_text", "")
        if not query:
            updates["errors"] = updates["errors"] + ["Retrieval skipped: empty query."]
            updates["retrieved_chunks"] = []
            updates["retrieval_end"] = time.time()
            return {**state, **updates}  # type: ignore[return-value]

        try:
            result = retriever.retrieve(query, top_k=top_k)
            updates["retrieved_chunks"] = result["results"]
        except Exception as exc:
            error_msg = f"Retrieval failed: {exc}"
            logger.error(error_msg)
            updates["errors"] = updates["errors"] + [error_msg]
            updates["retrieved_chunks"] = []

        updates["retrieval_end"] = time.time()
        return {**state, **updates}  # type: ignore[return-value]

    return retrieval_node


def make_generation_node():
    """Factory for the generation node."""

    def generation_node(state: PipelineState) -> PipelineState:
        updates: dict[str, Any] = {
            "generation_start": time.time(),
            "errors": state.get("errors", []),
        }

        query = state.get("transcribed_text", "")
        chunks = state.get("retrieved_chunks", [])

        if not query or not chunks:
            updates["answer"] = FALLBACK_RESPONSE
            updates["generation_end"] = time.time()
            return {**state, **updates}  # type: ignore[return-value]

        try:
            answer = generate_answer(query, chunks)
            updates["answer"] = answer
        except GenerationError as exc:
            error_msg = f"Generation failed: {exc}"
            logger.error(error_msg)
            updates["errors"] = updates["errors"] + [error_msg]
            updates["answer"] = FALLBACK_RESPONSE
        except Exception as exc:
            error_msg = f"Generation unexpected error: {exc}"
            logger.exception(error_msg)
            updates["errors"] = updates["errors"] + [error_msg]
            updates["answer"] = FALLBACK_RESPONSE

        updates["generation_end"] = time.time()
        return {**state, **updates}  # type: ignore[return-value]

    return generation_node


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_rag_graph(retriever: HybridRetriever, top_k: int = TOP_K) -> Any:
    """
    Compile and return the LangGraph StateGraph for the RAG pipeline.

    Args:
        retriever: Initialised HybridRetriever instance.
        top_k: Number of chunks to retrieve.

    Returns:
        Compiled LangGraph app (callable with .invoke(state)).
    """
    graph = StateGraph(PipelineState)

    graph.add_node("stt", stt_node)
    graph.add_node("retrieval", make_retrieval_node(retriever, top_k=top_k))
    graph.add_node("generation", make_generation_node())

    graph.add_edge(START, "stt")
    graph.add_edge("stt", "retrieval")
    graph.add_edge("retrieval", "generation")
    graph.add_edge("generation", END)

    return graph.compile()


# ── RAGPipeline class (orchestrator) ─────────────────────────────────────────

class RAGPipeline:
    """
    High-level orchestrator wrapping the LangGraph compiled app.

    Usage:
        pipeline = RAGPipeline(retriever)
        result = pipeline.answer(audio_path="query.wav")
        result = pipeline.answer(text_query="what is deep learning?")
    """

    def __init__(self, retriever: HybridRetriever, top_k: int = TOP_K) -> None:
        self._app = build_rag_graph(retriever, top_k=top_k)
        self._top_k = top_k

    def answer(
        self,
        audio_path: str | None = None,
        text_query: str | None = None,
    ) -> dict[str, Any]:
        """
        Run the full pipeline and return a structured result dict.

        Args:
            audio_path: Path to audio file for STT.
            text_query: Direct text query (bypasses STT).

        Returns:
            Dict:
              query:            original query (audio_path or text_query)
              transcribed_text: STT output (or text_query)
              retrieved_chunks: list of retrieval results
              answer:           LLM-generated answer
              stage_timings:    {stt_ms, retrieval_ms, generation_ms, total_ms}
              errors:           list of error strings (empty if successful)
        """
        pipeline_start = time.time()

        initial_state: PipelineState = {
            "audio_path": audio_path,
            "text_query": text_query,
            "errors": [],
        }

        final_state: PipelineState = self._app.invoke(initial_state)

        # ── Compute timings ───────────────────────────────────────────────────
        def _ms(start_key: str, end_key: str) -> float:
            s = final_state.get(start_key, 0.0)  # type: ignore[arg-type]
            e = final_state.get(end_key, 0.0)    # type: ignore[arg-type]
            return round((e - s) * 1000, 2) if s and e else 0.0

        pipeline_end = time.time()
        stt_ms = _ms("stt_start", "stt_end")
        retrieval_ms = _ms("retrieval_start", "retrieval_end")
        generation_ms = _ms("generation_start", "generation_end")
        total_ms = round((pipeline_end - pipeline_start) * 1000, 2)

        return {
            "query": audio_path or text_query or "",
            "transcribed_text": final_state.get("transcribed_text", ""),
            "retrieved_chunks": final_state.get("retrieved_chunks", []),
            "answer": final_state.get("answer", FALLBACK_RESPONSE),
            "stage_timings": {
                "stt_ms": stt_ms,
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms": total_ms,
            },
            "errors": final_state.get("errors", []),
        }
