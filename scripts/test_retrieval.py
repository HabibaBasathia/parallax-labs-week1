"""
Tests semantic search retrieval performance and logs latency per query.
"""

import time
from src.embeddings import embed_chunks
from src.vector_store import get_collection

TEST_QUERIES = [
    "a story about a friendly dog",
    "what happened at the park",
    "a character learns a lesson",
]


def run_query(collection, query: str, top_k: int = 3):
    start = time.perf_counter()

    query_embedding, _, _ = embed_chunks([query])
    results = collection.query(
        query_embeddings=[query_embedding[0].tolist()],
        n_results=top_k,
    )

    elapsed = time.perf_counter() - start
    return results, elapsed


def main():
    collection = get_collection()

    for query in TEST_QUERIES:
        results, elapsed = run_query(collection, query)
        print(f"\nQuery: '{query}'  |  latency: {elapsed*1000:.1f} ms")
        docs = results.get("documents", [[]])[0]
        for i, doc in enumerate(docs):
            print(f"  [{i}] {doc[:80]}...")


if __name__ == "__main__":
    main()