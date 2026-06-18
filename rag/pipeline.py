"""
RAG pipeline: embeds code-review examples into ChromaDB and retrieves
the top-k most similar examples for a given code snippet.

Build the index:
    python rag/pipeline.py --build

Quick retrieval smoke test:
    python rag/pipeline.py --query "def foo(): pass"
"""

import argparse
import re
from pathlib import Path
from typing import Iterator

import chromadb
from chromadb.config import Settings
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EMBED_MODEL = "all-MiniLM-L6-v2"
CHROMA_DIR = str(Path(__file__).parent.parent / "data" / "chroma")
COLLECTION_NAME = "code_reviews"
DATASET_ID = "alenphilip/Code-Review-Assistant"
BATCH_SIZE = 512
DEFAULT_TOP_K = 3

# ---------------------------------------------------------------------------
# Dataset parsing
# ---------------------------------------------------------------------------

_USER_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.DOTALL)
_ASST_RE = re.compile(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", re.DOTALL)


def _parse_example(text: str) -> tuple[str, str] | None:
    """Extract (code_snippet, review) from a chat-formatted training example."""
    user_m = _USER_RE.search(text)
    asst_m = _ASST_RE.search(text)
    if not user_m or not asst_m:
        return None
    return user_m.group(1).strip(), asst_m.group(1).strip()


def _iter_corpus(limit: int | None = None) -> Iterator[tuple[str, str, str]]:
    """Yield (doc_id, code_snippet, review) from the corpus dataset."""
    ds = load_dataset(DATASET_ID, split="train")
    if limit:
        ds = ds.select(range(limit))
    for i, row in enumerate(ds):
        parsed = _parse_example(row["text"])
        if parsed:
            code, review = parsed
            yield str(i), code, review


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

class CodeReviewRAG:
    def __init__(self, embed_model: str = EMBED_MODEL, chroma_dir: str = CHROMA_DIR):
        self._embedder = SentenceTransformer(embed_model)
        self._client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def build_index(self, limit: int | None = None) -> None:
        """Embed corpus and upsert into ChromaDB. Safe to call multiple times."""
        ids, codes, reviews = [], [], []
        for doc_id, code, review in _iter_corpus(limit=limit):
            ids.append(doc_id)
            codes.append(code)
            reviews.append(review)

        total = len(ids)
        print(f"Embedding {total} examples in batches of {BATCH_SIZE}…")
        for start in range(0, total, BATCH_SIZE):
            batch_ids = ids[start : start + BATCH_SIZE]
            batch_codes = codes[start : start + BATCH_SIZE]
            batch_reviews = reviews[start : start + BATCH_SIZE]
            embeddings = self._embedder.encode(batch_codes, show_progress_bar=False).tolist()
            self._collection.upsert(
                ids=batch_ids,
                embeddings=embeddings,
                documents=batch_codes,
                metadatas=[{"review": r} for r in batch_reviews],
            )
            pct = min(start + BATCH_SIZE, total)
            print(f"  {pct}/{total}")

        print(f"Index built — {self._collection.count()} documents in '{COLLECTION_NAME}'.")

    def retrieve(self, code: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        """
        Return the top-k most similar (code, review) examples for a query snippet.

        Each result dict has keys: code, review, distance.
        """
        query_embedding = self._embedder.encode([code]).tolist()
        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            out.append({"code": doc, "review": meta["review"], "distance": dist})
        return out

    def format_context(self, examples: list[dict]) -> str:
        """Format retrieved examples as a prompt context block."""
        parts = []
        for i, ex in enumerate(examples, 1):
            parts.append(
                f"### Example {i}\n"
                f"**Code:**\n```\n{ex['code']}\n```\n"
                f"**Review:**\n{ex['review']}"
            )
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAG pipeline for code review")
    parser.add_argument("--build", action="store_true", help="Build/refresh the ChromaDB index")
    parser.add_argument("--limit", type=int, default=None, help="Cap corpus size (for smoke tests)")
    parser.add_argument("--query", type=str, default=None, help="Retrieve examples for a code snippet")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    rag = CodeReviewRAG()

    if args.build:
        rag.build_index(limit=args.limit)

    if args.query:
        results = rag.retrieve(args.query, top_k=args.top_k)
        for i, r in enumerate(results, 1):
            print(f"\n--- Result {i} (distance={r['distance']:.4f}) ---")
            print(r["code"][:300])
            print("Review:", r["review"][:300])


if __name__ == "__main__":
    main()
