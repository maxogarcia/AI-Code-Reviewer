"""
FastAPI backend for the AI Code Reviewer.

Start with:
    uvicorn api.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.model import engine
from api.schemas import ReviewRequest, ReviewResponse, RetrievedExample
from rag.pipeline import CodeReviewRAG

rag: CodeReviewRAG | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    rag = CodeReviewRAG()
    yield


app = FastAPI(
    title="AI Code Reviewer",
    description="QLoRA fine-tuned Qwen2.5-Coder + RAG for context-aware code review.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/review", response_model=ReviewResponse)
def review(req: ReviewRequest) -> ReviewResponse:
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG index not ready")

    examples = rag.retrieve(req.code, top_k=req.top_k)
    context = rag.format_context(examples)

    review_text = engine.generate(
        code=req.code,
        context=context,
        max_new_tokens=req.max_new_tokens,
    )

    return ReviewResponse(
        review=review_text,
        retrieved_examples=[RetrievedExample(**ex) for ex in examples],
    )
