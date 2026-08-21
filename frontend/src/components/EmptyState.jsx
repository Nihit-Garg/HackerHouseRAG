import { Link } from "react-router-dom"

export default function EmptyState({ icon: Icon, title, text, linkTo, linkText }) {
  return (
    <div className="empty-state">
      <div className="empty-state-icon">
        <Icon size={22} strokeWidth={2} />
      </div>
      <h3>{title}</h3>
      <p>{text}</p>
      {linkTo && (
        <Link className="empty-state-link" to={linkTo}>
          {linkText}
        </Link>
      )}
    </div>
  )
}
