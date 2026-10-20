"""
ChromaDB setup + ingestion of chunks/embeddings.
Handles a few common edge cases: empty chunks, duplicate IDs, and
mismatched embedding/chunk counts.
"""

from typing import TYPE_CHECKING, Any, List, Sequence

import chromadb
from chromadb.config import Settings

if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection

DB_PATH: str = "data/chroma_db"
COLLECTION_NAME: str = "week2_corpus"


def get_collection() -> "Collection":
    """
    Open the persistent ChromaDB database and return the corpus collection.
    The collection is created if it does not exist yet.
    """
    client = chromadb.PersistentClient(path=DB_PATH, settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection


def ingest(
    collection: "Collection",
    chunks: List[str],
    embeddings: Sequence[Any],
    source_doc_id: str,
) -> int:
    """
    Store chunks and their embeddings in the collection.

    Empty chunks are skipped. Chunk IDs look like "doc_0_chunk_1", and
    upsert is used so running the pipeline twice does not create duplicates.

    Args:
        collection: The ChromaDB collection to write to.
        chunks: The chunk texts.
        embeddings: One embedding per chunk, in the same order.
        source_doc_id: ID of the story the chunks came from.

    Returns:
        The number of chunks actually stored.

    Raises:
        ValueError: If the number of chunks and embeddings differ.
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