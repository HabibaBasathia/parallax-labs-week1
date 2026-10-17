"""
FastAPI app that wraps the RAG system.
Endpoints: GET /health, POST /search, POST /ask
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.embeddings import embed_chunks
from src.rag_pipeline import answer_question
from src.vector_store import get_collection

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rag_api")

VALID_SENTIMENTS = {"positive", "negative", "neutral"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the embedding model once at startup, so the first request is not slow
    logger.info("Warming up embedding model...")
    embed_chunks(["warm up"])
    yield


app = FastAPI(title="RAG API", lifespan=lifespan)


# ---------- Request logging ----------

@app.middleware("http")
async def log_requests(request: Request, call_next):
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
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(3, ge=1, le=20)
    sentiment: Optional[str] = None
    topic: Optional[str] = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


# ---------- Helpers ----------

def build_where(sentiment: Optional[str], topic: Optional[str]):
    """Turns the optional filters into a ChromaDB 'where' filter."""
    conditions = []
    if sentiment:
        conditions.append({"sentiment": sentiment})
    if topic:
        conditions.append({"topic": topic})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def run_search(query: str, top_k: int, where):
    """Embeds the query and searches ChromaDB. Returns a list of results."""
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
def health():
    return {"status": "ok"}


@app.post("/search")
def search(req: SearchRequest):
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
def ask(req: AskRequest):
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