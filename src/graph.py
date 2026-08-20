"""
graph.py — LangGraph state graph for the RAG pipeline with Day 2 guardrails.

Graph flow with conditional edges:
  START
    → stt_node                 (transcribe or pass text)
    → unsafe_input_guard_node  (blocklist check) ──[blocked]──→ END
    → off_topic_guard_node     (embedding distance) ──[off-topic]──→ END
    → retrieval_node           (hybrid dense+BM25)
    → retrieval_threshold_node (skip LLM if weak results) ──[low]──→ END
    → generation_node          (Ollama with grounding prompt)
    → grounding_check_node     (parse grounding JSON + term overlap)
    → END

All guardrail rejections set state['guardrail_triggered'] and state['guardrail_detail']
for auditability. Errors in any node accumulate in state['errors'], never thrown.
"""
from __future__ import annotations

import logging
import time
from typing import Annotated, Any, Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from config import TOP_K
from generation import FALLBACK_RESPONSE, GenerationError
from guardrails import (
    check_grounding,
    check_off_topic,
    check_retrieval_threshold,
    check_unsafe_input,
    parse_grounding_json,
)
from harness import generation_stage, retrieval_stage
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
    query_embedding: Any           # np.ndarray | None — computed once, reused
    retrieved_chunks: list[dict[str, Any]]
    answer: str

    # Timing (unix timestamps)
    stt_start: float
    stt_end: float
    retrieval_start: float
    retrieval_end: float
    generation_start: float
    generation_end: float

    # Guardrails
    guardrail_triggered: str | None   # None or name of triggered guard
    guardrail_detail: dict[str, Any]  # detail dict from guard result

    # Error list — accumulated, never thrown
    errors: list[str]


# ── Routing helpers ───────────────────────────────────────────────────────────

def _route_guard(state: PipelineState) -> Literal["continue", "end"]:
    """Conditional edge: if a guardrail was triggered, jump to END."""
    if state.get("guardrail_triggered"):
        return "end"
    return "continue"


# ── Node implementations ───────────────────────────────────────────────────────

def stt_node(state: PipelineState) -> PipelineState:
    """Transcribe audio → text, or pass text_query through."""
    updates: dict[str, Any] = {
        "stt_start": time.time(),
        "errors": state.get("errors", []),
        "guardrail_triggered": state.get("guardrail_triggered"),
        "guardrail_detail": state.get("guardrail_detail", {}),
    }

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


def make_unsafe_input_guard_node():
    """Guard 1: keyword blocklist check on transcribed query."""

    def unsafe_input_guard_node(state: PipelineState) -> PipelineState:
        query = state.get("transcribed_text", "")
        # Empty query = STT failure or no input — not an unsafe content issue.
        # Let retrieval/threshold nodes handle empty queries downstream.
        if not query or not query.strip():
            return state

        guard_result = check_unsafe_input(query)

        if not guard_result["passed"]:
            logger.warning("Guard 1 (unsafe_input) triggered: %s", guard_result["reason"])
            return {
                **state,
                "guardrail_triggered": "unsafe_input",
                "guardrail_detail": guard_result,
                "answer": "I'm unable to process this request.",
                "retrieved_chunks": [],
            }  # type: ignore[return-value]

        return state  # pass-through

    return unsafe_input_guard_node


def make_off_topic_guard_node(embedder: Any):
    """Guard 2: embedding distance from corpus centroid."""

    def off_topic_guard_node(state: PipelineState) -> PipelineState:
        query = state.get("transcribed_text", "")
        if not query:
            return state

        # Embed query (reuse if already computed)
        q_emb = state.get("query_embedding")
        if q_emb is None:
            try:
                q_emb = embedder.embed_query(query)
            except Exception as exc:
                logger.warning("Off-topic guard: embedding failed (%s) — skipping.", exc)
                return state

        guard_result = check_off_topic(q_emb)

        if not guard_result["passed"]:
            logger.info("Guard 2 (off_topic) triggered: %s", guard_result["reason"])
            return {
                **state,
                "query_embedding": q_emb,
                "guardrail_triggered": "off_topic",
                "guardrail_detail": guard_result,
                "answer": "Your question appears to be outside the scope of this knowledge base.",
                "retrieved_chunks": [],
            }  # type: ignore[return-value]

        return {**state, "query_embedding": q_emb}  # type: ignore[return-value]

    return off_topic_guard_node


def make_retrieval_node(retriever: HybridRetriever, top_k: int = TOP_K):
    """Hybrid dense+BM25 retrieval via harness with input validation."""

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

        stage_result = retrieval_stage(query, retriever, top_k=top_k)
        if stage_result["ok"]:
            updates["retrieved_chunks"] = stage_result["output"]["results"]
        else:
            updates["errors"] = updates["errors"] + [f"Retrieval failed: {stage_result['error']}"]
            updates["retrieved_chunks"] = []

        updates["retrieval_end"] = time.time()
        return {**state, **updates}  # type: ignore[return-value]

    return retrieval_node


def make_retrieval_threshold_guard_node():
    """Guard 3: skip LLM if best RRF score is too weak."""

    def retrieval_threshold_guard_node(state: PipelineState) -> PipelineState:
        chunks = state.get("retrieved_chunks", [])
        guard_result = check_retrieval_threshold(chunks)

        if not guard_result["passed"]:
            logger.info("Guard 3 (retrieval_threshold) triggered: %s", guard_result["reason"])
            return {
                **state,
                "guardrail_triggered": "low_retrieval",
                "guardrail_detail": guard_result,
                "answer": "I don't have enough relevant information in my knowledge base to answer that question.",
            }  # type: ignore[return-value]

        return state

    return retrieval_threshold_guard_node


def make_generation_node():
    """Generation via harness with retry. Uses grounding-aware system prompt."""

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

        stage_result = generation_stage(query, chunks)
        if stage_result["ok"]:
            updates["answer"] = stage_result["output"]
        else:
            updates["errors"] = updates["errors"] + [f"Generation failed: {stage_result['error']}"]
            updates["answer"] = FALLBACK_RESPONSE

        updates["generation_end"] = time.time()
        return {**state, **updates}  # type: ignore[return-value]

    return generation_node


def make_grounding_check_node():
    """Guard 4: post-generation grounding verification."""

    def grounding_check_node(state: PipelineState) -> PipelineState:
        raw_answer = state.get("answer", "")
        chunks = state.get("retrieved_chunks", [])

        guard_result = check_grounding(raw_answer, chunks)
        clean_answer = guard_result["detail"].get("clean_answer", raw_answer)

        if not guard_result["passed"]:
            logger.warning("Guard 4 (grounding) triggered: %s", guard_result["reason"])
            return {
                **state,
                "answer": FALLBACK_RESPONSE,
                "guardrail_triggered": state.get("guardrail_triggered") or "ungrounded",
                "guardrail_detail": guard_result,
            }  # type: ignore[return-value]

        # Even if passed, strip the JSON block from the final answer
        return {
            **state,
            "answer": clean_answer,
            "guardrail_triggered": state.get("guardrail_triggered"),
            "guardrail_detail": guard_result,
        }  # type: ignore[return-value]

    return grounding_check_node


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_rag_graph(retriever: HybridRetriever, embedder: Any, top_k: int = TOP_K) -> Any:
    """
    Compile the LangGraph StateGraph with all guardrail nodes.

    New flow vs Day 1:
      stt → unsafe_guard → off_topic_guard → retrieval → threshold_guard
          → generation → grounding_check → END

    Conditional edges short-circuit to END on any guard failure.
    """
    graph = StateGraph(PipelineState)

    # Register nodes
    graph.add_node("stt", stt_node)
    graph.add_node("unsafe_input_guard", make_unsafe_input_guard_node())
    graph.add_node("off_topic_guard", make_off_topic_guard_node(embedder))
    graph.add_node("retrieval", make_retrieval_node(retriever, top_k=top_k))
    graph.add_node("retrieval_threshold_guard", make_retrieval_threshold_guard_node())
    graph.add_node("generation", make_generation_node())
    graph.add_node("grounding_check", make_grounding_check_node())

    # Linear edges
    graph.add_edge(START, "stt")
    graph.add_edge("stt", "unsafe_input_guard")

    # Conditional edges — short-circuit to END if guard triggered
    graph.add_conditional_edges(
        "unsafe_input_guard",
        _route_guard,
        {"continue": "off_topic_guard", "end": END},
    )
    graph.add_conditional_edges(
        "off_topic_guard",
        _route_guard,
        {"continue": "retrieval", "end": END},
    )
    graph.add_edge("retrieval", "retrieval_threshold_guard")
    graph.add_conditional_edges(
        "retrieval_threshold_guard",
        _route_guard,
        {"continue": "generation", "end": END},
    )
    graph.add_edge("generation", "grounding_check")
    graph.add_edge("grounding_check", END)

    return graph.compile()


# ── RAGPipeline class (orchestrator) ─────────────────────────────────────────

class RAGPipeline:
    """
    High-level orchestrator wrapping the LangGraph compiled app.

    Usage:
        pipeline = RAGPipeline(retriever, embedder)
        result = pipeline.answer(audio_path="query.wav")
        result = pipeline.answer(text_query="what is deep learning?")
    """

    def __init__(self, retriever: HybridRetriever, embedder: Any, top_k: int = TOP_K) -> None:
        self._app = build_rag_graph(retriever, embedder, top_k=top_k)
        self._top_k = top_k

    def answer(
        self,
        audio_path: str | None = None,
        text_query: str | None = None,
    ) -> dict[str, Any]:
        """
        Run the full pipeline (with guardrails) and return a structured result.

        Returns:
            Dict:
              query:               original input
              transcribed_text:    STT output or text_query
              retrieved_chunks:    list of retrieval results
              answer:              final answer (clean, JSON-stripped)
              stage_timings:       {stt_ms, retrieval_ms, generation_ms, total_ms}
              guardrail_triggered: None | "unsafe_input" | "off_topic" | "low_retrieval" | "ungrounded"
              guardrail_detail:    detail dict from the triggered guard (or {})
              errors:              list of error strings
        """
        pipeline_start = time.time()

        initial_state: PipelineState = {
            "audio_path": audio_path,
            "text_query": text_query,
            "errors": [],
            "guardrail_triggered": None,
            "guardrail_detail": {},
        }

        final_state: PipelineState = self._app.invoke(initial_state)

        def _ms(start_key: str, end_key: str) -> float:
            s = final_state.get(start_key, 0.0)  # type: ignore[arg-type]
            e = final_state.get(end_key, 0.0)    # type: ignore[arg-type]
            return round((e - s) * 1000, 2) if s and e else 0.0

        pipeline_end = time.time()

        return {
            "query": audio_path or text_query or "",
            "transcribed_text": final_state.get("transcribed_text", ""),
            "retrieved_chunks": final_state.get("retrieved_chunks", []),
            "answer": final_state.get("answer", FALLBACK_RESPONSE),
            "stage_timings": {
                "stt_ms": _ms("stt_start", "stt_end"),
                "retrieval_ms": _ms("retrieval_start", "retrieval_end"),
                "generation_ms": _ms("generation_start", "generation_end"),
                "total_ms": round((pipeline_end - pipeline_start) * 1000, 2),
            },
            "guardrail_triggered": final_state.get("guardrail_triggered"),
            "guardrail_detail": final_state.get("guardrail_detail", {}),
            "errors": final_state.get("errors", []),
        }
