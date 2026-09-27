"""
Embedding generation using sentence-transformers, with timing logged
so we can report embedding throughput in the README.
"""

import time
from typing import List
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good baseline

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_chunks(chunks: List[str]):
    """
    Returns (embeddings, elapsed_seconds, seconds_per_chunk).
    """
    model = get_model()

    start = time.perf_counter()
    embeddings = model.encode(chunks, show_progress_bar=False)
    elapsed = time.perf_counter() - start

    per_chunk = elapsed / len(chunks) if chunks else 0.0
    print(f"Embedded {len(chunks)} chunks in {elapsed:.2f}s ({per_chunk*1000:.1f} ms/chunk)")

    return embeddings, elapsed, per_chunk