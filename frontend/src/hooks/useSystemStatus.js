import { useCallback, useEffect, useState } from "react"
import { getHealth, getIndexStatus } from "../api"

export function useSystemStatus() {
  const [status, setStatus] = useState({
    loading: true,
    healthy: false,
    indexReady: false,
    totalChunks: null,
  })

  const refresh = useCallback(async () => {
    setStatus((prev) => ({ ...prev, loading: true }))

    const [healthResult, indexResult] = await Promise.allSettled([getHealth(), getIndexStatus()])

    setStatus({
      loading: false,
      healthy: healthResult.status === "fulfilled",
      indexReady:
        indexResult.status === "fulfilled" &&
        indexResult.value.faiss_index_exists &&
        indexResult.value.bm25_index_exists,
      totalChunks: indexResult.status === "fulfilled" ? indexResult.value.total_chunks ?? null : null,
    })
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  return { ...status, refresh }
}
