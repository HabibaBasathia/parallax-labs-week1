"""
End-to-end RAG pipeline: embed query -> retrieve chunks -> check
relevance (catches out-of-domain questions before calling the LLM) ->
build prompt -> call LLM -> log latency at each stage.
"""

import time
from src.embeddings import embed_chunks
from src.vector_store import get_collection
from src.prompt_template import SYSTEM_PROMPT, build_user_prompt
from src.llm_client import call_llm

TOP_K = 3
# ChromaDB's default distance is L2 (squared euclidean); LOWER = more similar.
# Anything above this threshold is treated as "not relevant enough" -> out of domain.
DISTANCE_THRESHOLD = 1.3


def answer_question(question: str) -> dict:
    """
    Runs the full RAG pipeline for one question.
    Returns a dict with the answer, retrieved chunks, and timing breakdown.
    """
    overall_start = time.perf_counter()

    # 1. Retrieve
    retrieval_start = time.perf_counter()
    collection = get_collection()
    query_embedding, _, _ = embed_chunks([question])
    results = collection.query(
        query_embeddings=[query_embedding[0].tolist()],
        n_results=TOP_K,
    )
    retrieval_time = time.perf_counter() - retrieval_start

    retrieved_docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    # 2. Out-of-domain check: if nothing was retrieved, or the closest
    #    match is still too far away, don't bother calling the LLM.
    is_out_of_domain = (
        not retrieved_docs
        or not distances
        or min(distances) > DISTANCE_THRESHOLD
    )

    if is_out_of_domain:
        return {
            "answer": "I don't have enough information in the provided context to answer this question.",
            "retrieved_chunks": retrieved_docs,
            "out_of_domain": True,
            "retrieval_time_s": round(retrieval_time, 3),
            "generation_time_s": 0.0,
            "total_time_s": round(time.perf_counter() - overall_start, 3),
        }

    # 3. Build prompt + call LLM
    generation_start = time.perf_counter()
    user_prompt = build_user_prompt(question, retrieved_docs)
    answer = call_llm(SYSTEM_PROMPT, user_prompt)
    generation_time = time.perf_counter() - generation_start

    total_time = time.perf_counter() - overall_start

    return {
        "answer": answer,
        "retrieved_chunks": retrieved_docs,
        "out_of_domain": False,
        "retrieval_time_s": round(retrieval_time, 3),
        "generation_time_s": round(generation_time, 3),
        "total_time_s": round(total_time, 3),
    }