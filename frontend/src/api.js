const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000"

async function request(path, options) {
  let response
  try {
    response = await fetch(`${BASE_URL}${path}`, options)
  } catch {
    throw new Error("Can't reach the Lumina server. Is it running?")
  }

  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new Error(body?.detail || `Request failed with status ${response.status}`)
  }

  return response.json()
}

export function getHealth() {
  return request("/health")
}

export function getIndexStatus() {
  return request("/index/status")
}

export function askText(query, topK) {
  return request("/query/text", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: topK }),
  })
}

export function askAudio(file) {
  const formData = new FormData()
  formData.append("file", file, file.name)
  return request("/query/audio", { method: "POST", body: formData })
}
