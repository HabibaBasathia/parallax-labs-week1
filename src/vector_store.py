"""
ChromaDB setup + ingestion of chunks/embeddings.
Handles a few common edge cases: empty chunks, duplicate IDs, and
mismatched embedding/chunk counts.
"""

import chromadb
from chromadb.config import Settings

DB_PATH = "data/chroma_db"
COLLECTION_NAME = "week2_corpus"


def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH, settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection


def ingest(collection, chunks, embeddings, source_doc_id: str):
    """
    Ingest chunks + their embeddings into the collection.
    Skips empty chunks and guards against count mismatches.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
        )

    ids, docs, embs, metas = [], [], [], []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        if not chunk or not chunk.strip():
            continue  # edge case: skip empty chunk
        chunk_id = f"{source_doc_id}_chunk_{i}"
        ids.append(chunk_id)
        docs.append(chunk)
        embs.append(emb.tolist() if hasattr(emb, "tolist") else emb)
        metas.append({"source_doc_id": source_doc_id, "chunk_index": i})

    if not ids:
        return 0  # edge case: nothing valid to ingest

    collection.upsert(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
    return len(ids)