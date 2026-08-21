import { Activity, BarChart3, Cpu, Database, RefreshCw } from "lucide-react"
import { useHistory } from "../hooks/useHistory"
import { useSystemStatus } from "../hooks/useSystemStatus"
import { formatMs, formatShortTime, truncate } from "../utils/format"
import IconCircle from "../components/IconCircle"

const PIPELINE_STAGES = [
  { key: "sttMs", label: "Transcription" },
  { key: "retrievalMs", label: "Search" },
  { key: "generationMs", label: "Generation" },
]

export default function StatusPage() {
  const { loading, healthy, indexReady, totalChunks, refresh } = useSystemStatus()
  const { entries, clearHistory } = useHistory()

  const latest = entries[0]
  const recent = entries.slice(0, 4)

  function handleClear() {
    if (confirm("Clear your entire question history? This can't be undone.")) {
      clearHistory()
    }
  }

  return (
    <div>
      <div className="page-header">
        <div className="page-heading">
          <IconCircle icon={BarChart3} />
          <h1>System Status</h1>
        </div>
        <button className="ghost-button" onClick={refresh} disabled={loading}>
          <RefreshCw size={14} strokeWidth={2.4} />
          Refresh
        </button>
      </div>

      <p className="page-subtitle">Live health and performance of the Lumina pipeline.</p>

      <div className="status-grid">
        <div className="status-stat">
          <span className="status-stat-label">
            <Activity size={15} strokeWidth={2.2} />
            Core Engine
          </span>
          <span className="status-stat-value">
            {loading ? "—" : healthy ? "Optimal" : "Offline"}
            <span className={`status-pill ${healthy ? "success" : "danger"}`}>
              <span className="status-dot" />
              {healthy ? "Online" : "Unreachable"}
            </span>
          </span>
        </div>

        <div className="status-stat">
          <span className="status-stat-label">
            <Database size={15} strokeWidth={2.2} />
            Knowledge Base
          </span>
          <span className="status-stat-value">
            {totalChunks ?? "—"}
            <span className={`status-pill ${indexReady ? "success" : "warning"}`}>
              <span className="status-dot" />
              {indexReady ? "Indexed" : "Not built"}
            </span>
          </span>
        </div>

        <div className="status-stat">
          <span className="status-stat-label">
            <Cpu size={15} strokeWidth={2.2} />
            Local LLM
          </span>
          <span className="status-stat-value">
            Enabled
            <span className="status-pill neutral">
              <span className="status-dot" />
              Runs locally
            </span>
          </span>
        </div>
      </div>

      <div className="status-columns">
        <div className="card">
          <div className="section-heading-row">
            <h2>Latest Query Pipeline</h2>
          </div>

          {latest ? (
            <div>
              <p className="pipeline-total">Total elapsed time: {formatMs(latest.timings.totalMs)}</p>
              {PIPELINE_STAGES.map(({ key, label }) => {
                const value = latest.timings[key]
                const total = latest.timings.totalMs
                const percent = total > 0 ? Math.min(100, (value / total) * 100) : 0

                return (
                  <div className="pipeline-row" key={key}>
                    <span className="pipeline-ms">{formatMs(value)}</span>
                    <span className="pipeline-track">
                      <span className="pipeline-fill" style={{ width: `${percent}%` }} />
                    </span>
                    <span className="pipeline-label">{label}</span>
                  </div>
                )
              })}
            </div>
          ) : (
            <p className="muted-note">Run a query to see pipeline timing.</p>
          )}
        </div>

        <div className="card">
          <div className="section-heading-row">
            <h2>Session History</h2>
            {recent.length > 0 && (
              <button className="link-button" onClick={handleClear}>
                Clear
              </button>
            )}
          </div>

          {recent.length === 0 ? (
            <p className="muted-note">No recent activity.</p>
          ) : (
            <div className="session-list">
              {recent.map((entry) => (
                <div className="session-item" key={entry.id}>
                  <div className="session-time">{formatShortTime(entry.timestamp)}</div>
                  <div className="session-question">"{truncate(entry.question, 70)}"</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
