"""
Adds NLP metadata (topic + sentiment) to every chunk in ChromaDB.
Run this once after run_pipeline.py.
"""

from collections import Counter

from src.vector_store import get_collection
from src.nlp_analysis import (
    get_sentiment,
    fit_topic_model,
    assign_topics,
)

BATCH_SIZE = 1000  # update ChromaDB in small batches


def main():
    collection = get_collection()

    # 1. Read all chunks
    data = collection.get(include=["documents", "metadatas"])
    ids = data["ids"]
    texts = data["documents"]
    old_metadatas = data["metadatas"]
    print(f"Loaded {len(ids)} chunks from ChromaDB.")

    if not ids:
        print("No chunks found. Run python run_pipeline.py first.")
        return

    # 2. Learn topics
    print("Learning topics (this can take 1-2 minutes)...")
    vectorizer, lda, topic_names = fit_topic_model(texts)
    print("\nTopics found:")
    for i, name in enumerate(topic_names):
        print(f"  Topic {i}: {name}")

    # 3. Label every chunk
    topics = assign_topics(texts, vectorizer, lda, topic_names)
    sentiments = [get_sentiment(t) for t in texts]

    # 4. Save labels back into ChromaDB (keep the old metadata too)
    for start in range(0, len(ids), BATCH_SIZE):
        end = start + BATCH_SIZE
        new_metadatas = []
        for i in range(start, min(end, len(ids))):
            merged = dict(old_metadatas[i] or {})
            merged["topic"] = topics[i]
            merged["sentiment"] = sentiments[i]
            new_metadatas.append(merged)

        collection.update(ids=ids[start:end], metadatas=new_metadatas)

    # 5. Summary
    print("\nDone! Labels saved to ChromaDB.")
    print("\nChunks per topic:")
    for name, count in Counter(topics).most_common():
        print(f"  {name}: {count}")
    print("\nChunks per sentiment:")
    for name, count in Counter(sentiments).most_common():
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()