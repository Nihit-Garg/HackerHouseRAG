import { ShieldAlert, ShieldCheck } from "lucide-react"
import { describeOutcome } from "../utils/guardrails"
import { formatConfidence } from "../utils/format"
import SourceChip from "./SourceChip"

export default function AnswerCard({ entry }) {
  const outcome = describeOutcome(entry.guardrailTriggered)
  const confidence = formatConfidence(entry.confidence)
  const Icon = outcome.tone === "success" ? ShieldCheck : ShieldAlert

  return (
    <div className="answer-card">
      <div className="answer-card-header">
        <span className={`status-pill ${outcome.tone}`}>
          <Icon size={13} strokeWidth={2.4} />
          {outcome.label}
        </span>
        {confidence && <span className="answer-card-meta">Confidence: {confidence}</span>}
      </div>

      <div className="answer-card-body">{entry.answer}</div>

      {entry.sources.length > 0 && (
        <div className="answer-sources">
          {entry.sources.map((source) => (
            <SourceChip key={source.id} source={source} />
          ))}
        </div>
      )}
    </div>
  )
}
