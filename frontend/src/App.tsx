import { useState } from "react";
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

const PLACEHOLDER = `# Paste your code here, e.g.:
def login(user, password):
    query = "SELECT * FROM users WHERE name = '" + user + "'"
    db.execute(query)`;

export default function App() {
  const [code, setCode] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ReviewResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  async function handleSubmit() {
    if (!code.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setExpandedIdx(null);

    try {
      const res = await fetch(`${API_BASE}/review`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? `HTTP ${res.status}`);
      }
      setResult(await res.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  function toggleExample(i: number) {
    setExpandedIdx(expandedIdx === i ? null : i);
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>AI Code Reviewer</h1>
        <p className="subtitle">QLoRA fine-tuned Qwen2.5-Coder + RAG</p>
      </header>

      <main className="app-main">
        <section className="editor-section">
          <label htmlFor="code-input">Paste your code</label>
          <textarea
            id="code-input"
            value={code}
            onChange={(e) => setCode(e.target.value)}
            placeholder={PLACEHOLDER}
            spellCheck={false}
          />
          <button
            className="submit-btn"
            onClick={handleSubmit}
            disabled={loading || !code.trim()}
          >
            {loading ? "Reviewing…" : "Review Code"}
          </button>
        </section>

        {error && <div className="error-banner">{error}</div>}

        {result && (
          <section className="results">
            <div className="review-box">
              <h2>Review</h2>
              <pre className="review-text">{result.review}</pre>
            </div>

            {result.retrieved_examples.length > 0 && (
              <div className="examples">
                <h2>
                  Retrieved context ({result.retrieved_examples.length}{" "}
                  examples)
                </h2>
                {result.retrieved_examples.map((ex, i) => (
                  <div key={i} className="example-card">
                    <button
                      className="example-toggle"
                      onClick={() => toggleExample(i)}
                    >
                      <span>Example {i + 1}</span>
                      <span className="distance">
                        similarity {(1 - ex.distance).toFixed(3)}
                      </span>
                      <span className="chevron">
                        {expandedIdx === i ? "▲" : "▼"}
                      </span>
                    </button>
                    {expandedIdx === i && (
                      <div className="example-body">
                        <h4>Code</h4>
                        <pre>{ex.code}</pre>
                        <h4>Review</h4>
                        <pre>{ex.review}</pre>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </section>
        )}
      </main>
    </div>
  );
}
