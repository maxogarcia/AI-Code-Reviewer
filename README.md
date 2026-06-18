# AI Code Reviewer

An LLM-powered code review assistant built end-to-end across six phases — from fine-tuning a 7B model to a full-stack web app with RAG, a REST API, and CI/CD. Paste any code snippet and get instant feedback on bugs, security issues, and style.

---

## Preview

```
┌──────────────────────────────────────────────────────────────────────┐
│  [<>] AI Code Reviewer                  Qwen2.5-Coder · QLoRA · RAG │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│                                                                       │
│                           ┌────────┐                                  │
│                           │  < >   │                                  │
│                           └────────┘                                  │
│                                                                       │
│                       AI Code Reviewer                                │
│           Paste any code snippet below and get instant                │
│              feedback on bugs, security issues, and style.            │
│                                                                       │
│                                                                       │
│                                                                       │
├──────────────────────────────────────────────────────────────────────┤
│   Paste code here…                                          [ ► ]    │
│                        Ctrl+Enter to review                           │
└──────────────────────────────────────────────────────────────────────┘
```

After submitting code, the chat interface shows a user bubble with your snippet, an animated loading indicator, and the model's review alongside the retrieved RAG examples:

```
┌──────────────────────────────────────────────────────────────────────┐
│  [<>] AI Code Reviewer                  Qwen2.5-Coder · QLoRA · RAG │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   You                                                                 │
│   ┌──────────────────────────────────────────────┐                   │
│   │ def login(user, password):                   │                   │
│   │     q = "SELECT * FROM users WHERE name="+user│                  │
│   │     db.execute(q)                            │                   │
│   └──────────────────────────────────────────────┘                   │
│                                                                       │
│  [●] AI Code Reviewer                                                 │
│   Critical Issue: SQL Injection Vulnerability                         │
│                                                                       │
│   The query is built with string concatenation, which allows an      │
│   attacker to inject arbitrary SQL. Use parameterised queries:        │
│                                                                       │
│     db.execute("SELECT * FROM users WHERE name = ?", (user,))        │
│                                                                       │
│   ▸ Retrieved context · 3 examples                                    │
│                                                                       │
├──────────────────────────────────────────────────────────────────────┤
│   Paste code here…                                          [ ► ]    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Project phases

This project was built in six sequential phases:

### Phase 1 — Repo scaffold & environment
Set up the project structure, Python virtual environment (Python 3.13), CUDA-enabled PyTorch, and verified the fine-tuning dataset loads correctly from Hugging Face.

### Phase 2 — QLoRA fine-tuning
Fine-tuned **Qwen2.5-Coder-7B-Instruct** on 13,670 code-review examples from [`alenphilip/Code-Review-Assistant`](https://huggingface.co/datasets/alenphilip/Code-Review-Assistant) using QLoRA (rank 16, 4-bit NF4 quantization via BitsAndBytes). Training was tracked with MLflow. The resulting adapter is saved to `outputs/qlora-adapter/`.

### Phase 3 — RAG pipeline
Embedded the same 13,670-example corpus using `all-MiniLM-L6-v2` (sentence-transformers) and stored the vectors in a local ChromaDB collection. At review time, the top-3 most similar examples are retrieved by cosine similarity and prepended to the prompt as few-shot context.

### Phase 4 — FastAPI backend
Built a REST API with two endpoints — `GET /health` and `POST /review`. The inference engine lazy-loads the QLoRA adapter on first request (so the server starts instantly) and is kept in memory as a thread-safe singleton for all subsequent requests.

### Phase 5 — React frontend
Built a dark-themed chat interface with Vite + React + TypeScript. Code is submitted via a pinned bottom input bar (Ctrl+Enter or the send button). The response renders as a chat conversation with a bot avatar, animated loading dots, and a collapsible section showing the retrieved RAG examples with similarity scores.

### Phase 6 — Docker & CI/CD
Containerised both services with separate Dockerfiles and a `docker-compose.yml` that reserves a GPU for the API container. GitHub Actions runs on every push to `main`: Python linting with `ruff`, TypeScript type-checking, a production Vite build, and a Docker image build with layer caching.

---

## How it works

```
Your code
   │
   ├─► RAG Pipeline (ChromaDB + sentence-transformers)
   │       └─ retrieves the 3 most similar code-review examples
   │
   └─► Fine-tuned LLM (Qwen2.5-Coder-7B + QLoRA adapter)
           └─ generates the review, conditioned on retrieved context
```

---

## Requirements

| Requirement | Notes |
|---|---|
| Python | 3.13 |
| CUDA | 12.1+ |
| GPU VRAM | ≥ 8 GB (model runs in 4-bit NF4 quantization) |
| Node.js | 22+ |

> CPU-only is not supported. Any CUDA-capable GPU with 8 GB+ VRAM will work.

---

## Quick start

```bash
# 1. Clone and enter the repo
git clone https://github.com/your-username/ai-code-reviewer.git
cd ai-code-reviewer

# 2. Create and activate a Python virtual environment
python3.13 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Install Python dependencies
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==5.12.1 peft==0.19.1 trl==1.6.0 accelerate==1.14.0 \
            bitsandbytes==0.49.2 datasets==5.0.0 \
            chromadb==1.5.9 sentence-transformers==5.6.0 \
            fastapi==0.137.1 "uvicorn[standard]==0.49.0" \
            mlflow==3.14.0

# 4. Build the RAG index (one-time, ~5 minutes)
python rag/pipeline.py --build

# 5. Start the API  (Terminal 1)
uvicorn api.main:app --port 8000

# 6. Start the frontend  (Terminal 2)
cd frontend && npm install && npm run dev
```

Open **http://localhost:5173** — paste any code and hit Ctrl+Enter.

> The base model (`Qwen/Qwen2.5-Coder-7B-Instruct`, ~15 GB) downloads from Hugging Face automatically on the first request. The **first review will take 30–60 seconds** while the model loads into VRAM; all subsequent reviews in the same session are fast (5–15 seconds).

---

## Performance

| Event | Typical time |
|---|---|
| Cold start (first request, model loading) | 30–60 s |
| RAG retrieval | ~0.5 s |
| Review generation at 512 max tokens | 5–15 s |
| Review generation at 256 max tokens | 3–7 s |

To trade response length for speed, pass a lower `max_new_tokens` value:

```bash
curl -s -X POST http://localhost:8000/review \
  -H "Content-Type: application/json" \
  -d '{"code": "...", "max_new_tokens": 256}' \
  | python3 -m json.tool
```

---

## What works well

- **Algorithm problems** — paste a LeetCode-style function; the model flags O(n²) loops, missing edge cases, and poor naming
- **Security issues** — SQL injection, hardcoded secrets, missing input validation, use of `eval` on user input
- **Python / JavaScript functions** — naming, structure, error handling, type hints, return types
- **Small classes or modules** — constructor design, single-responsibility, docstrings

**Tips:**
- Paste a single function or class rather than a full file for the sharpest feedback
- Add a comment at the top of the snippet to steer the review toward a specific concern
- The retrieved examples shown in the UI are the corpus examples that shaped the response — inspect them to understand why the model focused on what it did

---

## (Optional) Re-run fine-tuning

A trained adapter is already included in `outputs/qlora-adapter/`, so this step is only needed if you want to retrain from scratch.

```bash
# Full run (~3 hours)
python training/train.py

# Smoke test — 10 steps on 100 examples
python training/train.py --smoke-test

# View metrics
mlflow ui --port 5000
```

---

## Running with Docker

Requires [Docker](https://docs.docker.com/get-docker/) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
# Build the RAG index on the host first (one-time)
python rag/pipeline.py --build

# Start both services
docker compose up --build
```

- API: http://localhost:8000
- Frontend: http://localhost:80

---

## API reference

### `GET /health`
Returns `{"status": "ok"}`.

### `POST /review`

| Field | Type | Default | Description |
|---|---|---|---|
| `code` | string | required | Code snippet to review |
| `top_k` | int | `3` | RAG examples to retrieve (1–10) |
| `max_new_tokens` | int | `512` | Max review length (64–2048) |

**Response**
```json
{
  "review": "Critical Issue: SQL Injection...",
  "retrieved_examples": [
    { "code": "...", "review": "...", "distance": 0.27 }
  ]
}
```

---

## Project structure

```
ai-code-reviewer/
├── training/
│   └── train.py              # Phase 2 — QLoRA fine-tuning script
├── rag/
│   └── pipeline.py           # Phase 3 — ChromaDB index builder + retriever
├── api/
│   ├── main.py               # Phase 4 — FastAPI app (/health, /review)
│   ├── model.py              #           Lazy-loaded inference engine
│   └── schemas.py            #           Pydantic request/response models
├── frontend/
│   └── src/
│       ├── App.tsx            # Phase 5 — Chat UI
│       └── App.css            #           Styles
├── outputs/
│   └── qlora-adapter/        # Trained LoRA weights + tokenizer
├── data/
│   └── chroma/               # ChromaDB vector index (git-ignored, rebuilt by --build)
├── docker/
│   ├── Dockerfile.api        # Phase 6 — API container
│   ├── Dockerfile.frontend   #           Nginx-served frontend container
│   └── nginx.conf            #           SPA routing + API proxy
├── docker-compose.yml
└── .github/workflows/ci.yml  # Phase 6 — Lint, typecheck, Docker build
```

---

## CI

GitHub Actions runs on every push to `main`:

1. **Python lint** — `ruff check` over `api/`, `rag/`, `training/`
2. **Frontend typecheck + build** — `tsc --noEmit` + `vite build`
3. **Docker image build** — frontend container built with GHA layer cache

---

## License

MIT
