function isSameDay(a, b) {
  return (
    a.getFullYear() === b.getFullYear() &&
    a.getMonth() === b.getMonth() &&
    a.getDate() === b.getDate()
  )
}

export function formatTimestamp(epochMs) {
  const date = new Date(epochMs)
  const now = new Date()
  const yesterday = new Date(now)
  yesterday.setDate(now.getDate() - 1)

  const time = date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })

  if (isSameDay(date, now)) return `Today, ${time}`
  if (isSameDay(date, yesterday)) return `Yesterday, ${time}`
  return `${date.toLocaleDateString([], { month: "short", day: "numeric" })}, ${time}`
}

export function formatShortTime(epochMs) {
  return new Date(epochMs).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })
}

export function truncate(text, length = 90) {
  if (!text) return ""
  return text.length > length ? `${text.slice(0, length).trim()}…` : text
}

export function formatConfidence(value) {
  if (typeof value !== "number") return null
  return `${Math.round(value * 100)}%`
}

export function formatMs(value) {
  return `${Math.round(value || 0)}ms`
}

export function formatDuration(totalSeconds) {
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`
}
