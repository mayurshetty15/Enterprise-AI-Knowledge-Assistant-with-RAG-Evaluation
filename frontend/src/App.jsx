import { useCallback, useEffect, useRef, useState } from "react";

import EvaluationDashboard from "./components/EvaluationDashboard.jsx";
import { api } from "./services/api.js";

function StatusPill({ label, status }) {
  const ready = status === "ok" || status === "available";
  const text = status === "available" ? "Available" : status === "ok" ? "Online" : status || "Unknown";
  return (
    <span className={`status-pill ${ready ? "is-ready" : "is-warning"}`}>
      <span className="status-dot" aria-hidden="true" />
      {label}: {text}
    </span>
  );
}

function Evidence({ sources }) {
  if (!sources?.length) {
    return <p className="no-evidence">No supporting document evidence was retrieved for this response.</p>;
  }
  return (
    <div className="evidence-list">
      {sources.map((source) => (
        <details className="evidence-card" key={`${source.document_id}-${source.page}-${source.snippet}`}>
          <summary>
            <span className="file-mark" aria-hidden="true">PDF</span>
            <span>{source.filename}</span>
            <span className="page-label">Page {source.page}</span>
          </summary>
          <p>{source.snippet}</p>
        </details>
      ))}
    </div>
  );
}

function App() {
  const [activeView, setActiveView] = useState("assistant");
  const [health, setHealth] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [selectedDocuments, setSelectedDocuments] = useState([]);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);
  const chatEndRef = useRef(null);

  const refreshDocuments = useCallback(async () => {
    const result = await api.documents();
    setDocuments(result.documents);
    setSelectedDocuments((current) => current.filter((id) =>
      result.documents.some((document) => document.document_id === id)));
  }, []);

  const refreshHealth = useCallback(async () => {
    try {
      setHealth(await api.health());
    } catch {
      setHealth(null);
    }
  }, []);

  useEffect(() => {
    refreshDocuments().catch((requestError) => setError(requestError.message));
    refreshHealth();
    const timer = window.setInterval(refreshHealth, 30000);
    return () => window.clearInterval(timer);
  }, [refreshDocuments, refreshHealth]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, loading]);

  async function uploadFile(file) {
    if (!file) return;
    setError("");
    setNotice("");
    setUploading(true);
    try {
      const result = await api.upload(file);
      await refreshDocuments();
      setNotice(result.already_indexed
        ? `${result.filename} was already in the knowledge base.`
        : `${result.filename} indexed: ${result.chunk_count} text chunks across ${result.page_count} pages.`);
      await refreshHealth();
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setUploading(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  async function deleteDocument(document) {
    if (!window.confirm(`Delete ${document.filename} and its indexed content?`)) return;
    setError("");
    setNotice("");
    try {
      await api.removeDocument(document.document_id);
      setSelectedDocuments((current) => current.filter((id) => id !== document.document_id));
      await refreshDocuments();
      setNotice(`${document.filename} was deleted.`);
    } catch (requestError) {
      setError(requestError.message);
    }
  }

  async function sendQuestion(event) {
    event.preventDefault();
    const text = question.trim();
    if (!text || loading) return;
    setQuestion("");
    setError("");
    setMessages((current) => [...current, { role: "user", text }]);
    setLoading(true);
    try {
      const result = await api.ask(text, selectedDocuments);
      setMessages((current) => [...current, {
        role: "assistant",
        text: result.answer,
        sources: result.sources,
        status: result.status,
      }]);
    } catch (requestError) {
      setError(requestError.message);
      setMessages((current) => [...current, {
        role: "assistant",
        text: "I could not complete that request. Check the error notice and try again.",
        sources: [],
      }]);
    } finally {
      setLoading(false);
    }
  }

  function toggleDocument(documentId) {
    setSelectedDocuments((current) => current.includes(documentId)
      ? current.filter((id) => id !== documentId)
      : [...current, documentId]);
  }

  const serviceStatus = health?.services || {};
  const ollamaStatus = serviceStatus.ollama?.status || "unavailable";
  const modelStatus = serviceStatus.chat_model?.status || "unavailable";

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="Enterprise AI Knowledge Assistant home">
          <span className="brand-icon" aria-hidden="true">K</span>
          <span><strong>Knowledge</strong><small>LOCAL AI WORKSPACE</small></span>
        </a>
        <div className="topbar-right">
          <nav className="view-tabs" aria-label="Application sections">
            <button
              className={activeView === "assistant" ? "view-tab is-active" : "view-tab"}
              type="button"
              onClick={() => setActiveView("assistant")}
              aria-current={activeView === "assistant" ? "page" : undefined}
            >Assistant</button>
            <button
              className={activeView === "evaluation" ? "view-tab is-active" : "view-tab"}
              type="button"
              onClick={() => setActiveView("evaluation")}
              aria-current={activeView === "evaluation" ? "page" : undefined}
            >Evaluation</button>
          </nav>
          <span className="local-badge"><span aria-hidden="true">●</span> Local &amp; private</span>
          <button className="icon-button" type="button" onClick={refreshHealth} aria-label="Refresh service status">↻</button>
        </div>
      </header>

      <main>
        <section className="page-heading">
          <div>
            <p className="eyebrow">YOUR PRIVATE KNOWLEDGE BASE</p>
            <h1>Enterprise AI Knowledge Assistant</h1>
            <p className="subtitle">Ask questions and get answers grounded in your documents.</p>
          </div>
          <div className="health-row" aria-label="Service health">
            <StatusPill label="API" status={serviceStatus.api === "ok" ? "ok" : health ? "unavailable" : "unavailable"} />
            <StatusPill label="Ollama" status={ollamaStatus} />
            <StatusPill label="Chat model" status={modelStatus} />
          </div>
        </section>

        {(error || notice) && (
          <div className={`notice ${error ? "notice-error" : "notice-success"}`} role={error ? "alert" : "status"}>
            <span>{error || notice}</span>
            <button type="button" onClick={() => { setError(""); setNotice(""); }} aria-label="Dismiss message">×</button>
          </div>
        )}

        {activeView === "assistant" ? <div className="workspace">
          <aside className="sidebar">
            <section className="panel upload-panel">
              <div className="section-title">
                <div><span className="section-icon">↑</span><h2>Add documents</h2></div>
                <span className="subtle-label">PDF</span>
              </div>
              <div
                className={`dropzone ${dragging ? "is-dragging" : ""} ${uploading ? "is-uploading" : ""}`}
                onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={(event) => {
                  event.preventDefault();
                  setDragging(false);
                  uploadFile(event.dataTransfer.files[0]);
                }}
              >
                <span className="upload-art" aria-hidden="true">↑</span>
                <strong>{uploading ? "Processing your PDF…" : "Drop a PDF here"}</strong>
                <span>or choose a file from your device</span>
                <button className="secondary-button" type="button" disabled={uploading} onClick={() => inputRef.current?.click()}>
                  Choose PDF
                </button>
                <input
                  ref={inputRef}
                  className="visually-hidden"
                  type="file"
                  accept="application/pdf,.pdf"
                  onChange={(event) => uploadFile(event.target.files[0])}
                  aria-label="Choose a PDF to upload"
                />
                <small>PDF files only · Size limit configured by the API</small>
              </div>
            </section>

            <section className="panel documents-panel">
              <div className="section-title">
                <div><span className="section-icon">▤</span><h2>Knowledge base</h2></div>
                <span className="count-badge">{documents.length}</span>
              </div>
              <p className="panel-hint">Select documents to limit your search. Leave all unchecked to search everything.</p>
              {documents.length === 0 ? (
                <div className="empty-documents">
                  <span aria-hidden="true">▧</span>
                  <p>No documents yet</p>
                  <small>Upload a PDF to get started.</small>
                </div>
              ) : (
                <ul className="document-list">
                  {documents.map((document) => (
                    <li className="document-item" key={document.document_id}>
                      <label className="document-select">
                        <input
                          type="checkbox"
                          checked={selectedDocuments.includes(document.document_id)}
                          onChange={() => toggleDocument(document.document_id)}
                          aria-label={`Limit questions to ${document.filename}`}
                        />
                        <span className="file-mark" aria-hidden="true">PDF</span>
                        <span className="document-info">
                          <strong title={document.filename}>{document.filename}</strong>
                          <small>{document.page_count} pages · {document.chunk_count} chunks</small>
                          <small className={`document-status ${document.status === "indexed" ? "" : "status-failed"}`}>
                            {document.status === "indexed" ? "Indexed" : document.status}
                          </small>
                        </span>
                      </label>
                      <button
                        className="delete-button"
                        type="button"
                        onClick={() => deleteDocument(document)}
                        aria-label={`Delete ${document.filename}`}
                      >×</button>
                    </li>
                  ))}
                </ul>
              )}
            </section>
          </aside>

          <section className="panel chat-panel" aria-label="Question and answer chat">
            <div className="chat-heading">
              <div><span className="assistant-avatar" aria-hidden="true">✦</span><div><h2>Ask your knowledge base</h2><p>Responses are based on your indexed documents</p></div></div>
              <span className="model-tag">{serviceStatus.chat_model?.name || "Local model"}</span>
            </div>
            <div className="conversation" aria-live="polite">
              {messages.length === 0 ? (
                <div className="chat-empty">
                  <div className="empty-orbit" aria-hidden="true"><span>✦</span></div>
                  <h3>Your documents, ready to answer</h3>
                  <p>Ask a question about your uploaded PDFs. Answers include page-level evidence so you can verify every response.</p>
                  <div className="suggestion-card">
                    <span aria-hidden="true">↳</span> What are the main findings in my documents?
                  </div>
                </div>
              ) : (
                <div className="message-list">
                  {messages.map((message, index) => (
                    <article className={`message message-${message.role}`} key={`${index}-${message.role}`}>
                      <span className="message-avatar" aria-hidden="true">{message.role === "user" ? "You" : "✦"}</span>
                      <div className="message-content">
                        <strong>{message.role === "user" ? "You" : "Knowledge assistant"}</strong>
                        <p>{message.text}</p>
                        {message.role === "assistant" && <Evidence sources={message.sources} />}
                      </div>
                    </article>
                  ))}
                  {loading && <div className="thinking" role="status"><span className="spinner" /> Searching evidence and preparing an answer…</div>}
                  <div ref={chatEndRef} />
                </div>
              )}
            </div>
            <form className="question-form" onSubmit={sendQuestion}>
              <label className="visually-hidden" htmlFor="question">Ask a question</label>
              <textarea
                id="question"
                rows="1"
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    sendQuestion(event);
                  }
                }}
                placeholder="Ask a question about your documents…"
                disabled={loading}
              />
              <button className="send-button" type="submit" disabled={loading || !question.trim()} aria-label="Send question">
                <span aria-hidden="true">↑</span>
              </button>
              <div className="input-caption"><span>Answers are generated locally with your selected documents.</span><span>Enter to send · Shift + Enter for a new line</span></div>
            </form>
          </section>
        </div> : <EvaluationDashboard />}
      </main>
      <footer><span>Private by design. Your documents stay on this device.</span><span>Powered by local Ollama + ChromaDB</span></footer>
    </div>
  );
}

export default App;
