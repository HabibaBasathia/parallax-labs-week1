"""
End-to-end RAG pipeline: embed query -> retrieve chunks -> check
relevance (catches out-of-domain questions before calling the LLM) ->
build prompt -> call LLM -> log latency at each stage.
"""

import time
from typing import Any, Dict

from src.embeddings import embed_chunks
from src.llm_client import call_llm
from src.prompt_template import SYSTEM_PROMPT, build_user_prompt
from src.vector_store import get_collection

TOP_K: int = 3
# ChromaDB's default distance is L2 (squared euclidean); LOWER = more similar.
# Anything above this threshold is treated as "not relevant enough" -> out of domain.
DISTANCE_THRESHOLD: float = 1.3


def answer_question(question: str) -> Dict[str, Any]:
    """
    Run the full RAG pipeline for one question.

    Steps: embed the question, retrieve the top chunks, skip the LLM if
    even the best chunk is too far away (out of domain), otherwise build
    the prompt and ask the LLM.

    Args:
        question: The user's question.

    Returns:
        A dict with these keys:
            answer: The answer text (or the refusal message).
            retrieved_chunks: The chunk texts that were retrieved.
            out_of_domain: True if the LLM was skipped as off-topic.
            retrieval_time_s: Seconds spent on embedding and search.
            generation_time_s: Seconds spent on the LLM call (0 if skipped).
            total_time_s: Total seconds for the whole call.
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