"""
FastAPI app that wraps the RAG system.

Endpoints:
    GET  /health  - check that the service is alive
    POST /search  - semantic search with optional topic / sentiment filters
    POST /ask     - full RAG pipeline (retrieve, then generate an answer)

Run with: python -m uvicorn src.api:app --reload
Then open http://127.0.0.1:8000/docs to try the endpoints.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from src.embeddings import embed_chunks
from src.rag_pipeline import answer_question
from src.vector_store import get_collection

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rag_api")

VALID_SENTIMENTS = {"positive", "negative", "neutral"}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the embedding model once at startup, so the first request is not slow."""
    logger.info("Warming up embedding model...")
    embed_chunks(["warm up"])
    yield


app = FastAPI(title="RAG API", lifespan=lifespan)


# ---------- Request logging ----------

@app.middleware("http")
async def log_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Log one line per request: method, path, status code and duration."""
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "method=%s path=%s status=%s duration_ms=%.1f",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


# ---------- Request shapes (FastAPI returns 422 if these are wrong) ----------

class SearchRequest(BaseModel):
    """Body of POST /search."""

    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(3, ge=1, le=20)
    sentiment: Optional[str] = None
    topic: Optional[str] = None


class AskRequest(BaseModel):
    """Body of POST /ask."""

    question: str = Field(..., min_length=1, max_length=1000)


# ---------- Helpers ----------

def build_where(sentiment: Optional[str], topic: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Turn the optional filters into a ChromaDB "where" filter.

    Args:
        sentiment: "positive", "negative", "neutral", or None for no filter.
        topic: A topic name, or None for no filter.

    Returns:
        None if no filter is set, a single condition if one is set, or an
        "$and" of both conditions.
    """
    conditions: List[Dict[str, str]] = []
    if sentiment:
        conditions.append({"sentiment": sentiment})
    if topic:
        conditions.append({"topic": topic})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def run_search(
    query: str, top_k: int, where: Optional[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Embed the query and search ChromaDB.

    Args:
        query: The search text.
        top_k: How many chunks to return.
        where: Optional ChromaDB filter from build_where.

    Returns:
        A list of results, each with id, text, distance, topic and
        sentiment. A lower distance means a closer match.
    """
    collection = get_collection()
    query_embedding, _, _ = embed_chunks([query])
    results = collection.query(
        query_embeddings=[query_embedding[0].tolist()],
        n_results=top_k,
        where=where,
    )

    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    return [
        {
            "id": chunk_id,
            "text": doc,
            "distance": distance,
            "topic": (meta or {}).get("topic"),
            "sentiment": (meta or {}).get("sentiment"),
        }
        for chunk_id, doc, meta, distance in zip(ids, docs, metas, distances)
    ]


# ---------- Endpoints ----------

@app.get("/health")
def health() -> Dict[str, str]:
    """Return {"status": "ok"} if the service is running."""
    return {"status": "ok"}


@app.post("/search")
def search(req: SearchRequest) -> Dict[str, Any]:
    """
    Semantic search over the chunks, with optional topic / sentiment filters.

    Errors:
        400: blank query, or a sentiment that is not positive/negative/neutral.
        422: wrong or missing fields (handled by FastAPI).
        500: an internal error while searching.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be blank.")
    if req.sentiment and req.sentiment not in VALID_SENTIMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"sentiment must be one of: {sorted(VALID_SENTIMENTS)}",
        )

    try:
        start = time.perf_counter()
        results = run_search(req.query, req.top_k, build_where(req.sentiment, req.topic))
        elapsed = time.perf_counter() - start
    except Exception:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail="Internal error while searching.")

    return {
        "query": req.query,
        "count": len(results),
        "results": results,
        "latency_s": round(elapsed, 3),
    }


@app.post("/ask")
def ask(req: AskRequest) -> Dict[str, Any]:
    """
    Answer a question with the full RAG pipeline.

    Returns the answer, the retrieved chunks and the latency breakdown.

    Errors:
        400: blank question.
        422: wrong or missing fields (handled by FastAPI).
        500: an internal error, or the LLM call failed.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be blank.")

    try:
        result = answer_question(req.question)
    except Exception:
        logger.exception("Ask failed")
        raise HTTPException(status_code=500, detail="Internal error while answering.")

    # llm_client returns a string starting with "Error:" when the LLM call fails
    if result["answer"].startswith("Error:"):
        raise HTTPException(status_code=500, detail=result["answer"])

    return result