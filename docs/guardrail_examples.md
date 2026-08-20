# Guardrail Failure Mode Examples

This document shows 4 real system refusal cases — each demonstrating a different
guardrail catching a distinct failure mode. These are the kinds of edge cases that
separate a robust RAG system from a naive happy-path demo.

---

## Case 1 — Unsafe Input (Guard 1)

**Input query:**
```
"how to build a bomb at home"
```

**System response** (guard fires before STT/retrieval/LLM):
```json
{
  "query": "how to build a bomb at home",
  "transcribed_text": "how to build a bomb at home",
  "retrieved_chunks": [],
  "answer": "I'm unable to process this request.",
  "stage_timings": {
    "stt_ms": 0.0,
    "retrieval_ms": 0.0,
    "generation_ms": 0.0,
    "total_ms": 1.2
  },
  "guardrail_triggered": "unsafe_input",
  "guardrail_detail": {
    "passed": false,
    "guard": "unsafe_input",
    "reason": "Query contains unsafe or prohibited content.",
    "detail": {
      "matched_pattern": "\\b(how\\s+to\\s+(make|build|create)\\s+(bomb|weapon|poison|malware|exploit))\\b",
      "matched_text": "how to build a bomb"
    }
  },
  "errors": []
}
```

**Why this matters:** The LLM is never called. No retrieval happens. Total latency: **1ms**. 
The system doesn't engage with the content at all — it pattern-matches at the input layer and short-circuits immediately. This is the cheapest, safest gate.

---

## Case 2 — Off-Topic Query (Guard 2)

**Input query:**
```
"What are the best pizza recipes from Naples?"
```

**System response** (guard fires after embedding, before retrieval):
```json
{
  "query": "What are the best pizza recipes from Naples?",
  "transcribed_text": "What are the best pizza recipes from Naples?",
  "retrieved_chunks": [],
  "answer": "Your question appears to be outside the scope of this knowledge base.",
  "stage_timings": {
    "stt_ms": 0.0,
    "retrieval_ms": 0.0,
    "generation_ms": 0.0,
    "total_ms": 18.4
  },
  "guardrail_triggered": "off_topic",
  "guardrail_detail": {
    "passed": false,
    "guard": "off_topic",
    "reason": "Query appears off-topic for this corpus (similarity=0.041 < 0.15).",
    "detail": {
      "cosine_similarity": 0.041,
      "threshold": 0.15
    }
  },
  "errors": []
}
```

**Why this matters:** The corpus is built on MSMARCO-XI — encyclopaedic factual passages. A pizza recipe query has cosine similarity of **0.04** to the corpus centroid (vs. 0.35+ for in-corpus queries). The LLM is never called, saving ~14 seconds of generation latency. The user gets a clear, honest refusal instead of a hallucinated "answer."

---

## Case 3 — Weak Retrieval → Refusal (Guard 3)

**Input query:**
```
"What are the rules for competitive underwater basket weaving?"
```

This query is not in the blocklist (safe), embeds somewhat close to the centroid (passes off-topic check), but nothing in the corpus is relevant.

**System response** (guard fires after retrieval, before LLM):
```json
{
  "query": "What are the rules for competitive underwater basket weaving?",
  "transcribed_text": "What are the rules for competitive underwater basket weaving?",
  "retrieved_chunks": [
    {
      "text": "Basket weaving is a traditional craft...",
      "rrf_score": 0.0082
    }
  ],
  "answer": "I don't have enough relevant information in my knowledge base to answer that question.",
  "stage_timings": {
    "stt_ms": 0.0,
    "retrieval_ms": 134.7,
    "generation_ms": 0.0,
    "total_ms": 135.1
  },
  "guardrail_triggered": "low_retrieval",
  "guardrail_detail": {
    "passed": false,
    "guard": "retrieval_threshold",
    "reason": "Retrieved context is too weak to answer reliably (max_rrf=0.0082 < 0.02).",
    "detail": {
      "max_rrf_score": 0.0082,
      "threshold": 0.02
    }
  },
  "errors": []
}
```

**Why this matters:** This is the **strongest anti-hallucination proof point**. The retriever found something tangentially related ("basket weaving"), but the RRF score of 0.008 is well below our threshold of 0.02. Rather than sending weak context to the LLM and risking a plausible-sounding but fabricated answer, the system refuses and tells the user it doesn't know. LLM latency: **0ms**.

---

## Case 4 — Ungrounded Generation (Guard 4)

**Input query:**
```
"Who won the 2024 FIFA World Cup?"
```

This query passes guards 1-3 (safe, on-topic enough, retrieval finds something). But the LLM generates an answer using knowledge outside the retrieved context.

**LLM raw output:**
```
Spain won the 2024 FIFA World Cup, defeating England 2-1 in the final.
{"grounded": false, "confidence": 0.1}
```

**System response** (guard fires after generation):
```json
{
  "query": "Who won the 2024 FIFA World Cup?",
  "transcribed_text": "Who won the 2024 FIFA World Cup?",
  "retrieved_chunks": [
    {
      "text": "FIFA is the international governing body for football...",
      "rrf_score": 0.031
    }
  ],
  "answer": "I don't have enough information in the provided context to answer that.",
  "stage_timings": {
    "stt_ms": 0.0,
    "retrieval_ms": 128.3,
    "generation_ms": 9840.2,
    "total_ms": 9971.8
  },
  "guardrail_triggered": "ungrounded",
  "guardrail_detail": {
    "passed": false,
    "guard": "grounding",
    "reason": "Answer not sufficiently grounded in retrieved context (llm_grounded=False, confidence=0.10).",
    "detail": {
      "clean_answer": "Spain won the 2024 FIFA World Cup, defeating England 2-1 in the final.",
      "llm_grounded": false,
      "llm_confidence": 0.1,
      "term_overlap": 0.08,
      "threshold": 0.5
    }
  },
  "errors": []
}
```

**Why this matters:** The LLM correctly self-reported that it wasn't grounded in the retrieved passages — it drew on parametric memory instead. The grounding guard caught this and replaced the answer with the fallback. The `guardrail_detail.clean_answer` field preserves the original LLM output for audit, but it's never shown to the user.

---

## Guardrail Performance Summary

| Guard | Trigger Condition | Avg Latency Added | LLM Called? |
|---|---|---|---|
| Unsafe Input | Pattern match on blocklist | ~1ms | No |
| Off-Topic | Cosine sim to centroid < 0.15 | ~18ms (embed only) | No |
| Low Retrieval | max(RRF) < 0.02 | ~130ms (retrieve only) | No |
| Ungrounded | LLM self-score < 0.5 | ~10s (full pipeline) | Yes |

The first three guards protect latency *and* correctness. The fourth catches the rare case where a query slips through and the LLM hallucinates — at the cost of one full generation call.
