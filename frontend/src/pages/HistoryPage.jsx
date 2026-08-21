import { Bot, History, MessageCircleQuestion, ShieldAlert, ShieldCheck, User } from "lucide-react"
import { useHistory } from "../hooks/useHistory"
import { describeOutcome } from "../utils/guardrails"
import { formatTimestamp } from "../utils/format"
import IconCircle from "../components/IconCircle"
import SourceChip from "../components/SourceChip"
import EmptyState from "../components/EmptyState"

export default function HistoryPage() {
  const { entries, clearHistory } = useHistory()

  function handleClear() {
    if (confirm("Clear your entire question history? This can't be undone.")) {
      clearHistory()
    }
  }

  return (
    <div>
      <div className="page-header">
        <div className="page-heading">
          <IconCircle icon={History} />
          <h1>Interaction History</h1>
        </div>
        {entries.length > 0 && (
          <button className="ghost-button" onClick={handleClear}>
            Clear history
          </button>
        )}
      </div>

      <p className="page-subtitle">Review your past questions and the answers Lumina gave.</p>

      {entries.length === 0 ? (
        <EmptyState
          icon={MessageCircleQuestion}
          title="No questions yet"
          text="Everything you ask on the Ask page will show up here."
          linkTo="/ask"
          linkText="Ask a question"
        />
      ) : (
        <div className="history-list">
          {entries.map((entry) => {
            const outcome = describeOutcome(entry.guardrailTriggered)
            const StatusIcon = outcome.tone === "success" ? ShieldCheck : ShieldAlert

            return (
              <div className="history-card" key={entry.id}>
                <div className="history-question-row">
                  <div className="history-question">
                    <div className="history-avatar">
                      <User size={15} strokeWidth={2.2} />
                    </div>
                    <p>"{entry.question}"</p>
                  </div>
                  <span className="history-timestamp">{formatTimestamp(entry.timestamp)}</span>
                </div>

                <div className="history-answer-row">
                  <div className="history-avatar bot">
                    <Bot size={15} strokeWidth={2.2} />
                  </div>
                  <div className="history-answer-content">
                    <p>{entry.answer}</p>
                    <div className="history-tags">
                      {entry.sources.map((source) => (
                        <SourceChip key={source.id} source={source} />
                      ))}
                      <span className={`status-pill ${outcome.tone}`}>
                        <StatusIcon size={12} strokeWidth={2.4} />
                        {outcome.label}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
