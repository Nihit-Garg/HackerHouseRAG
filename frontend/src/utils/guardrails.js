export const OUTCOME_LABELS = {
  verified: { label: "Verified Response", tone: "success" },
  unsafe_input: { label: "Blocked — unsafe content", tone: "danger" },
  off_topic: { label: "Outside knowledge base", tone: "warning" },
  low_retrieval: { label: "Insufficient information", tone: "warning" },
  ungrounded: { label: "Couldn't be verified", tone: "warning" },
  error: { label: "Connection error", tone: "danger" },
}

export function describeOutcome(guardrailTriggered) {
  if (!guardrailTriggered) return OUTCOME_LABELS.verified
  return OUTCOME_LABELS[guardrailTriggered] || OUTCOME_LABELS.ungrounded
}
