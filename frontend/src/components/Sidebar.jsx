import { NavLink } from "react-router-dom"
import { BarChart3, History, Info, MessageCircle, ShieldCheck, Sparkles } from "lucide-react"
import { useSystemStatus } from "../hooks/useSystemStatus"

const NAV_ITEMS = [
  { to: "/about", label: "About", icon: Info },
  { to: "/ask", label: "Ask", icon: MessageCircle },
  { to: "/history", label: "History", icon: History },
  { to: "/status", label: "Status", icon: BarChart3 },
]

function statusTag({ loading, healthy, indexReady }) {
  if (loading) return "Checking status…"
  if (!healthy) return "Server unavailable"
  if (!indexReady) return "Knowledge base not built"
  return "Knowledge base ready"
}

export default function Sidebar() {
  const status = useSystemStatus()

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="sidebar-brand-mark">
          <Sparkles size={20} strokeWidth={2.2} />
        </div>
        <div>
          <div className="sidebar-brand-name">Lumina AI</div>
          <div className="sidebar-brand-tag">{statusTag(status)}</div>
        </div>
      </div>

      <nav className="sidebar-nav">
        {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) => `sidebar-nav-item${isActive ? " active" : ""}`}
          >
            <Icon size={17} strokeWidth={2.2} />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="sidebar-footer-icon">
          <ShieldCheck size={15} strokeWidth={2.2} />
        </div>
        <div>
          <div className="sidebar-footer-title">Secure Session</div>
          <div className="sidebar-footer-text">
            Your questions aren't stored on any server — history stays only on this device.
          </div>
        </div>
      </div>
    </aside>
  )
}
