import { useEffect, useRef, useState } from "react";
import "./App.css";

const API_BASE = "http://localhost:8000";

interface RetrievedExample {
  code: string;
  review: string;
  distance: number;
}

interface ReviewResponse {
  review: string;
  retrieved_examples: RetrievedExample[];
}

interface Turn {
  code: string;
  response: ReviewResponse;
}

// ── Icons ──────────────────────────────────────────────────────────────────

function IconCode() {
  return (
    <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
      <path d="M8.293 6.293a1 1 0 0 1 1.414 1.414L5.414 12l4.293 4.293a1 1 0 0 1-1.414 1.414l-5-5a1 1 0 0 1 0-1.414l5-5zm7.414 0a1 1 0 0 1 1.414 0l5 5a1 1 0 0 1 0 1.414l-5 5a1 1 0 0 1-1.414-1.414L19.586 12l-4.293-4.293a1 1 0 0 1 0-1.414z" />
    </svg>
  );
}

function IconSend() {
  return (
    <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
      <path d="M2.01 21 23 12 2.01 3 2 10l15 2-15 2z" />
    </svg>
  );
}

function IconBot() {
  return (
    <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
      <path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7H3a7 7 0 0 1 7-7h1V5.73A2 2 0 0 1 10 4a2 2 0 0 1 2-2zM7.5 14a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3zm9 0a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3zM3 21v-2h18v2H3z" />
    </svg>
  );
}

// ── Retrieved examples accordion ──────────────────────────────────────────

function RagExamples({ examples }: { examples: RetrievedExample[] }) {
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  if (examples.length === 0) return null;

  return (
    <div className="rag-section">
      <div className="rag-heading">Retrieved context · {examples.length} examples</div>
      {examples.map((ex, i) => (
        <div key={i} className="example-card">
          <button
            className="example-toggle"
            onClick={() => setExpandedIdx(expandedIdx === i ? null : i)}
          >
            <span>Example {i + 1}</span>
            <span className="ex-similarity">
              {(1 - ex.distance).toFixed(3)} similarity
            </span>
            <span className="ex-chevron">{expandedIdx === i ? "▲" : "▼"}</span>
          </button>
          {expandedIdx === i && (
            <div className="example-body">
              <div>
                <div className="example-block-label">Code</div>
                <pre>{ex.code}</pre>
              </div>
              <div>
                <div className="example-block-label">Review</div>
                <pre>{ex.review}</pre>
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

// ── Main ──────────────────────────────────────────────────────────────────

export default function App() {
  const [code, setCode] = useState("");
  const [loading, setLoading] = useState(false);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [error, setError] = useState<string | null>(null);

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }, [code]);

  // Scroll to bottom when new content arrives
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns, loading]);

  async function handleSubmit() {
    const trimmed = code.trim();
    if (!trimmed || loading) return;

    setLoading(true);
    setError(null);
    setCode("");

    try {
      const res = await fetch(`${API_BASE}/review`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: trimmed }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? `HTTP ${res.status}`);
      }
      const data: ReviewResponse = await res.json();
      setTurns((prev) => [...prev, { code: trimmed, response: data }]);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      handleSubmit();
    }
  }

  const empty = turns.length === 0 && !loading && !error;

  return (
    <div className="shell">
      {/* ── Topbar ── */}
      <header className="topbar">
        <div className="topbar-logo">
          <div className="topbar-icon">
            <IconCode />
          </div>
          AI Code Reviewer
        </div>
        <div className="topbar-badge">Qwen2.5-Coder · QLoRA · RAG</div>
      </header>

      {/* ── Chat area ── */}
      <div className="chat-area">
        {empty ? (
          <div className="welcome">
            <div className="welcome-icon">
              <IconCode />
            </div>
            <h2>AI Code Reviewer</h2>
            <p>
              Paste any code snippet below and get instant feedback on bugs,
              security issues, and style.
            </p>
          </div>
        ) : (
          <div className="messages">
            {turns.map((turn, i) => (
              <div key={i}>
                {/* User message */}
                <div className="msg-user">
                  <div className="msg-label">You</div>
                  <div className="msg-user-bubble">
                    <pre>{turn.code}</pre>
                  </div>
                </div>

                {/* Assistant message */}
                <div className="msg-assistant" style={{ marginTop: 20 }}>
                  <div className="assistant-avatar">
                    <IconBot />
                  </div>
                  <div className="msg-assistant-body">
                    <div className="msg-label">AI Code Reviewer</div>
                    <div className="review-text">{turn.response.review}</div>
                    <RagExamples examples={turn.response.retrieved_examples} />
                  </div>
                </div>
              </div>
            ))}

            {/* Loading indicator */}
            {loading && (
              <div className="msg-assistant">
                <div className="assistant-avatar">
                  <IconBot />
                </div>
                <div className="msg-assistant-body">
                  <div className="msg-label">AI Code Reviewer</div>
                  <div className="loading-dots">
                    <span /><span /><span />
                  </div>
                </div>
              </div>
            )}

            <div ref={bottomRef} />
          </div>
        )}

        {error && (
          <div className="error-msg">
            <div className="error-banner">{error}</div>
          </div>
        )}
      </div>

      {/* ── Input bar ── */}
      <footer className="input-bar">
        <div className="input-wrap">
          <textarea
            ref={textareaRef}
            value={code}
            onChange={(e) => setCode(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Paste code here…"
            rows={1}
            spellCheck={false}
          />
          <button
            className="send-btn"
            onClick={handleSubmit}
            disabled={loading || !code.trim()}
            title="Review (Ctrl+Enter)"
          >
            <IconSend />
          </button>
        </div>
        <div className="input-hint">Ctrl+Enter to review</div>
      </footer>
    </div>
  );
}
