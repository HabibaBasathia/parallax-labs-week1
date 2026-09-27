"""
Week 2 pipeline: takes Week 1's clean_corpus.jsonl and runs it through
chunking -> embedding -> ChromaDB ingestion.
"""

import json
from src.chunking import recursive_chunk
from src.embeddings import embed_chunks
from src.vector_store import get_collection, ingest

INPUT_FILE = "data/processed/clean_corpus.jsonl"  # <- Week 1's output


def main():
    collection = get_collection()

    total_chunks_ingested = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for doc_id, line in enumerate(f):
            data = json.loads(line)
            text = data.get("text", "")

            chunks = recursive_chunk(text, chunk_size=500, chunk_overlap=50)
            if not chunks:
                continue

            embeddings, elapsed, per_chunk = embed_chunks(chunks)
            n = ingest(collection, chunks, embeddings, source_doc_id=f"doc_{doc_id}")
            total_chunks_ingested += n

    print(f"Done. Ingested {total_chunks_ingested} chunks into ChromaDB.")


if __name__ == "__main__":
    main()