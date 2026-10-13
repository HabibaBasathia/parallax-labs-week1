"""
Shows filtered retrieval: search by meaning AND by NLP labels
(topic / sentiment). Logs the latency of each search.
"""

import time

from src.embeddings import embed_chunks
from src.vector_store import get_collection

QUERY = "a child is sad and cries"
TOP_K = 3


def search(collection, query, where=None):
    """Runs one search. 'where' is the filter (or None for no filter)."""
    start = time.perf_counter()

    query_embedding, _, _ = embed_chunks([query])
    results = collection.query(
        query_embeddings=[query_embedding[0].tolist()],
        n_results=TOP_K,
        where=where,
    )

    elapsed = time.perf_counter() - start
    return results, elapsed


def show(title, results, elapsed):
    print("\n" + "=" * 60)
    print(title)
    print(f"Latency: {elapsed * 1000:.1f} ms")

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    if not docs:
        print("  (no results match this filter)")
        return

    for doc, meta in zip(docs, metas):
        print(f"  [{meta.get('topic')} | {meta.get('sentiment')}] {doc[:90]}...")


def main():
    collection = get_collection()

    # Warm-up: the first search is slow (model loading), so run one that is not measured
    search(collection, "warm up")

    # Pick a real topic name from the data (skip 'uncategorized')
    all_metas = collection.get(include=["metadatas"])["metadatas"]
    topics = sorted(
        {m["topic"] for m in all_metas if m.get("topic") and m["topic"] != "uncategorized"}
    )
    if not topics:
        print("No topic labels found. Run python -m scripts.enrich_corpus first.")
        return
    demo_topic = topics[0]

    print(f"Query: '{QUERY}'")
    print(f"Demo topic used for filters: {demo_topic}")

    tests = [
        ("1) No filter", None),
        ("2) Only POSITIVE chunks", {"sentiment": "positive"}),
        ("3) Only NEGATIVE chunks", {"sentiment": "negative"}),
        (f"4) Only topic '{demo_topic}'", {"topic": demo_topic}),
        (
            f"5) Topic '{demo_topic}' AND negative",
            {"$and": [{"topic": demo_topic}, {"sentiment": "negative"}]},
        ),
        ("6) Filter that matches nothing", {"topic": "this_topic_does_not_exist"}),
    ]

    for title, where in tests:
        results, elapsed = search(collection, QUERY, where)
        show(title, results, elapsed)


if __name__ == "__main__":
    main()