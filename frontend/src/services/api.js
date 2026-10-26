const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000")
  .replace(/\/$/, "");

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, options);
  } catch {
    throw new Error("Could not reach the backend. Check that the API is running.");
  }

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof payload.detail === "string"
      ? payload.detail
      : `Request failed (${response.status}).`;
    throw new Error(detail);
  }
  return payload;
}

export const api = {
  health: () => request("/api/health"),
  documents: () => request("/api/documents"),
  upload: (file) => {
    const body = new FormData();
    body.append("file", file);
    return request("/api/documents/upload", { method: "POST", body });
  },
  removeDocument: (documentId) => request(
    `/api/documents/${encodeURIComponent(documentId)}`,
    { method: "DELETE" },
  ),
  ask: (question, documentIds) => request("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, document_ids: documentIds }),
  }),
  evaluationRuns: () => request("/api/evaluation/runs"),
  evaluationRun: (runId) => request(
    `/api/evaluation/runs/${encodeURIComponent(runId)}`,
  ),
};
