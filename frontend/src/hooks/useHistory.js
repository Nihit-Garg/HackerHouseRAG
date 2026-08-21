import { useCallback, useEffect, useState } from "react"

const STORAGE_KEY = "lumina.history"

function readStoredEntries() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

export function useHistory() {
  const [entries, setEntries] = useState(readStoredEntries)

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(entries))
    } catch {
      return
    }
  }, [entries])

  const addEntry = useCallback((entry) => {
    const withMeta = { ...entry, id: crypto.randomUUID(), timestamp: Date.now() }
    setEntries((prev) => [withMeta, ...prev])
    return withMeta
  }, [])

  const clearHistory = useCallback(() => {
    setEntries([])
  }, [])

  return { entries, addEntry, clearHistory }
}
