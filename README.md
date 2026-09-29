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


---

# Week 3: LLM Integration & Prompt Engineering

## What I built
This week wraps Week 2's retrieval pipeline with an LLM to generate actual answers:

1. **LLM client** (`src/llm_client.py`) — calls Groq's free API using
   `openai/gpt-oss-20b`, with automatic retries (up to 3 attempts) on rate
   limits and connection errors, and a capped max output of 512 tokens.
2. **Prompt template** (`src/prompt_template.py`) — a system prompt that
   restricts the model to only using the retrieved context, and explicitly
   instructs it to say "I don't have enough information..." when the
   context doesn't cover the question.
3. **RAG pipeline** (`src/rag_pipeline.py`) — the full flow: embed query →
   retrieve top-3 chunks from ChromaDB → check retrieval distance against
   a threshold to catch out-of-domain questions *before* calling the LLM →
   build prompt → call LLM → log retrieval/generation/total latency.
4. **Test script** (`scripts/test_rag.py`) — runs sample questions through
   the pipeline, including one intentionally off-topic question, to verify
   the hallucination guard works.

## Prompt engineering approach
- System prompt explicitly forbids using outside knowledge and gives an
  exact fallback phrase for insufficient context, rather than letting the
  model improvise a refusal (keeps behavior consistent and testable).
- Context chunks are labeled (`[Context 1]`, `[Context 2]`, ...) so the
  model can distinguish between separate retrieved passages.
- Temperature set to 0.2 to reduce creative drift and keep answers close
  to the source material.

## Hallucination / out-of-domain handling
Two layers of defense:
1. **Retrieval-level check** — if ChromaDB's closest match distance exceeds
   a threshold (1.3), the question is treated as out-of-domain and the LLM
   is never called (saves latency and avoids the model guessing).
2. **Prompt-level guard** — even when context is retrieved, the system
   prompt forces the model to only use that context and to explicitly
   decline if it's insufficient.

Verified with a test question ("What is the capital of France?") against
a story-text corpus — correctly returned the fallback message instead of
answering from general knowledge.

## Error handling
`llm_client.py` retries up to 3 times on rate limit or connection errors
(5s delay between attempts) before returning a clear error string instead
of crashing. This was validated for real during development: the originally
planned model (`llama-3.1-8b-instant`) was deprecated by Groq on
2026-08-16, and the API's 404 error was caught and surfaced clearly rather
than crashing the pipeline, prompting a switch to `openai/gpt-oss-20b`.

## How to run
```bash
venv\Scripts\activate
pip install -r requirements.txt

# Make sure .env contains: GROQ_API_KEY=your_key_here

# Run the full chunk -> embed -> ingest pipeline (Week 2, if not already run)
python run_pipeline.py

# Run the RAG test questions (retrieval + generation + latency)
python -m scripts.test_rag

# Run unit tests
python -m pytest tests/
```

## Results
- All in-domain test questions answered correctly using only retrieved context.
- Off-topic question correctly triggered the "insufficient information" fallback.
- Typical latency after model warm-up: ~0.03-0.05s retrieval, ~1-1.5s generation.
- All 7 unit tests passing.