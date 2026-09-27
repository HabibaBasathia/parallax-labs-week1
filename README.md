# Parallax Labs - Week 1: Environment, Data Ingestion & Preprocessing

This repository contains the Week 1 deliverables for the Parallax Labs machine learning engineering internship, focusing on setting up a robust development environment, acquiring a real-world text corpus, and building a modular text-cleaning pipeline backed by unit tests[cite: 1].

## Project Structure
- `scripts/verify_env.py`: Asserts environment stability, dependency imports, and hardware status[cite: 1].
- `src/preprocessing.py`: Modular text-cleaning functions handling HTML stripping, Unicode normalization, and whitespace cleanup[cite: 1].
- `tests/test_preprocessing.py`: Automated `pytest` suite validating cleaning functions[cite: 1].
- `data/raw/`: Local directory containing raw ingested data (git-ignored)[cite: 1].
- `data/processed/`: Local directory containing the final validated clean corpus (`clean_corpus.jsonl`)[cite: 1].

## How to Run

1. **Clone the repository and set up your virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   ---

# Week 2: Chunking, Embeddings & Vector DB

## What I built
This week takes the cleaned corpus from Week 1 (`data/processed/clean_corpus.jsonl`)
and builds a semantic search pipeline on top of it:

1. **Chunking** (`src/chunking.py`) — a recursive splitter that tries paragraph
   breaks first, then sentence breaks, then word breaks, keeping chunks under
   500 characters with 50 characters of overlap so context isn't lost at
   chunk boundaries.
2. **Embeddings** (`src/embeddings.py`) — generates vector embeddings for each
   chunk using `sentence-transformers` (`all-MiniLM-L6-v2`), logging embedding
   time per batch.
3. **Vector DB** (`src/vector_store.py`) — ingests chunks + embeddings into a
   persistent ChromaDB collection, with metadata linking each chunk back to
   its source document.
4. **Retrieval test** (`scripts/test_retrieval.py`) — runs test queries against
   the collection and logs retrieval latency per query.

## Chunking strategy
Recursive splitting with chunk_size=500 characters and chunk_overlap=50
characters. Paragraph breaks are tried first to keep whole ideas together;
if a piece is still too long, it falls back to splitting on sentences, then
words. The overlap prevents losing context right at a chunk boundary.

## Edge cases handled
- Empty or whitespace-only chunks are skipped before ingestion.
- Mismatched chunk/embedding counts raise an explicit error.
- Duplicate chunk IDs are handled via `upsert` rather than `add`.

## How to run
```bash
venv\Scripts\activate
pip install -r requirements.txt

# Run the full chunk -> embed -> ingest pipeline
python run_pipeline.py

# Test retrieval + measure latency
python -m scripts.test_retrieval

# Run unit tests
python -m pytest tests/
```

## Results
- 11,161 chunks generated and ingested from the Week 1 corpus.
- Embedding speed: ~20-25 ms per chunk.
- Retrieval latency: ~15-20 ms per query.
- All 7 unit tests passing (5 chunking, 2 preprocessing).