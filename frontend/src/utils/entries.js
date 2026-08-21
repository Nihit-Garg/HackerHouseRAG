export function buildEntry(result) {
  const detail = result.guardrail_detail?.detail || {}

  return {
    question: result.transcribed_text || result.query || "",
    answer: result.answer,
    guardrailTriggered: result.guardrail_triggered || null,
    confidence: typeof detail.llm_confidence === "number" ? detail.llm_confidence : null,
    sources: (result.retrieved_chunks || []).map((chunk, index) => ({
      id: chunk.metadata?.chunk_id || `${chunk.metadata?.passage_id || "source"}-${index}`,
      snippet: chunk.text,
    })),
    timings: {
      sttMs: result.stage_timings?.stt_ms || 0,
      retrievalMs: result.stage_timings?.retrieval_ms || 0,
      generationMs: result.stage_timings?.generation_ms || 0,
      totalMs: result.stage_timings?.total_ms || 0,
    },
  }
}
