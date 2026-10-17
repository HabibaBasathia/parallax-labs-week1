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


---

# Week 4: NLP Analysis (Topic & Sentiment)

## What I built
This week adds NLP labels (metadata) to every chunk in ChromaDB, so search
can be filtered by topic and sentiment:

1. **NLP helpers** (`src/nlp_analysis.py`) - sentiment with VADER, and topic
   modeling with LDA (scikit-learn, 8 topics, each named by its top 3 words).
2. **Enrichment script** (`scripts/enrich_corpus.py`) - labels all 11,161
   chunks and saves `topic` and `sentiment` into ChromaDB metadata, keeping
   the old metadata.
3. **Manual labeling** (`scripts/make_label_set.py`) - shows 20 random chunks
   so I can label them by hand (without seeing VADER's answer).
4. **Evaluation** (`scripts/evaluate_sentiment.py`) - compares VADER with my
   labels. Report saved in `evaluation/sentiment_report.txt`.
5. **Filtered retrieval** (`scripts/test_filtered_retrieval.py`) - searches
   by meaning plus filters (sentiment, topic, or both) and logs latency.
6. **Unit tests** (`tests/test_nlp_analysis.py`) - 7 tests for sentiment,
   topic assignment and edge cases.

## Why LDA and VADER
I chose LDA because BERTopic often fails to install on Windows and needs
heavy models. LDA is simple, fast and reproducible (fixed random seed).
VADER needs no model download and runs fast on thousands of chunks.

## Results

**Topics found:** little_day_man, said_mom_mommy, timmy_play_day,
bird_happy_day, tree_day_big, lily_ben_tom, lily_mom_said, happy_car_day.
The topics are mostly character names plus common story words. The corpus
is children's stories, where the same names and words repeat a lot, so LDA
groups by characters more than by true themes.

**Sentiment counts (11,161 chunks):** positive 9,532, negative 1,533,
neutral 96.

**Sentiment accuracy:** VADER matched my manual labels on 17 of 20 chunks
(**85%**). Details are in `evaluation/sentiment_report.txt`.

**Limits of the evaluation:**
- The test set is small (20 chunks), and 15 of them were positive, so a
  tool that always said "positive" would already score about 75%.
- All 3 mistakes were chunks I labeled negative. A chunk is a piece cut
  from the middle of a story, and it can sound happy on its own while the
  story around it is sad. VADER only reads the words in the chunk, so it
  cannot see that. For example, a chunk about a sad boy who "was starting
  to feel better" was labeled positive.

## Edge cases handled
- Chunks shorter than 5 words get the topic `uncategorized`.
- Chunks with only rare or unknown words (jargon) get `uncategorized`.
- Words that appear in fewer than 5 chunks are ignored when learning topics.
- Empty text gets the sentiment `neutral`.
- A filter that matches nothing returns an empty result and does not crash.

## Filtered retrieval and latency
Query used: "a child is sad and cries".

| Test | Latency |
|---|---|
| 1) No filter (first search, includes one-time startup) | 7784.8 ms |
| 2) Positive only | 67.1 ms |
| 3) Negative only | 28.5 ms |
| 4) One topic | 26.2 ms |
| 5) Topic and negative | 30.6 ms |
| 6) Filter with no match | 17.5 ms |

The 7.8 s for test 1 is a one-time startup cost (loading the embedding model
and database), not the cost of searching. Filtered searches took 17-67 ms,
so filtering is fast. I did not measure an unfiltered search after warm-up,
so I can't give an exact filtering overhead.

Filters work: the negative filter returned only negative chunks, and the
topic + negative filter returned only chunks with both labels.

## How to run
```bash
venv\Scripts\activate
pip install -r requirements.txt

# Week 2 pipeline, if not already run
python run_pipeline.py

# Add topic + sentiment labels to ChromaDB
python -m scripts.enrich_corpus

# Label 20 chunks by hand, then check VADER's accuracy
python -m scripts.make_label_set
python -m scripts.evaluate_sentiment

# Try filtered search
python -m scripts.test_filtered_retrieval

# Run all unit tests
python -m pytest tests/
```

---

# Week 5: Evaluation & API (FastAPI)

## What I built
1. **FastAPI service** (`src/api.py`) - the RAG system behind web endpoints:
   - `GET /health` - checks that the service is alive.
   - `POST /search` - semantic search with optional `topic` and `sentiment` filters (Week 4 metadata).
   - `POST /ask` - full RAG pipeline (retrieve, then generate), returns the answer and timings.
2. **Request logging** - middleware logs one line per request with the method, path, status code and duration.
3. **Error handling:**
   - **422** - wrong or missing input fields (handled automatically by Pydantic).
   - **400** - valid type but unacceptable input (blank query or question, unknown sentiment).
   - **500** - internal failure (database error, or the LLM call failed).
4. **Test set builder** (`scripts/make_testset.py`) - the LLM writes one question for each of 20 random chunks. Relevant chunks for a question = all chunks from the same story. Saved to `evaluation/testset.json` (19 questions; the LLM returned an empty reply for one).
5. **Retrieval evaluation** (`scripts/evaluate_retrieval.py`) - Precision@K and Recall@K. Report: `evaluation/retrieval_report.txt`.
6. **End-to-end evaluation** (`scripts/evaluate_end_to_end.py`) - runs the full pipeline, grades answers with an LLM grader, tests off-topic questions, and logs latency. Report: `evaluation/end_to_end_report.txt`.
7. **API tests** (`tests/test_api.py`) - 15 pytest tests. The database and LLM are replaced with fakes, so the tests are fast and can trigger a 500 on purpose.

## Retrieval results (19 questions)
| K | Precision@K | Recall@K | Source chunk found |
|---|---|---|---|
| 1 | 0.74 | 0.37 | 9/19 |
| 3 | 0.40 | 0.60 | 13/19 |
| 5 | 0.25 | 0.61 | 13/19 |

Average search time: 17.7 ms.

- The top result is from the correct story 74% of the time (Precision@1).
- Recall rises from 0.37 to 0.60 between K=1 and K=3, then stays flat at K=5. A larger K only adds noise, which supports using `TOP_K = 3`.
- Precision@5 is low partly because a story has only 2.4 relevant chunks on average, so it cannot go above about 0.48.
- 6 of 19 questions did not find their source chunk in the top 5. Many of these are general questions ("What did Ben find on the shelf?") and many stories share the same character names.

## End-to-end results (19 in-domain + 3 off-topic questions)
- Correct answers: **12/19 (63%)**. Refusals count as wrong here.
- Refused ("not enough information"): 4/19. Of the 15 questions it actually answered, 12 were correct (80%).
- Accuracy when the source chunk **was** retrieved: **11/13 (85%)**.
- Accuracy when the source chunk was **not** retrieved: **1/6 (17%)**.
- Off-topic questions refused: **3/3**.
- Latency: retrieval 0.059 s, generation 0.752 s, total **0.811 s** on average. Slowest question: 1.479 s.

**Main finding:** answer quality depends on retrieval. When search finds the right chunk, answers are mostly correct. When it misses, the system usually refuses (the hallucination guard works) or answers wrongly. Future work: better retrieval (larger chunks, re-ranking, combining metadata filters).

## Limits of the evaluation
- Small test set (19 questions).
- "Relevant" means the same story. A chunk from another story could also answer a general question, so real retrieval quality is probably a bit higher than measured.
- The questions are written by an LLM from a chunk, and answers are graded by an LLM that sees only the source chunk. Both can be too strict or too lenient. The grader was not checked by hand.

## How to run
```bash
venv\Scripts\activate
pip install -r requirements.txt

# Start the API, then open http://127.0.0.1:8000/docs to try the endpoints
python -m uvicorn src.api:app --reload

# Build the test set, then run both evaluations
python -m scripts.make_testset
python -m scripts.evaluate_retrieval
python -m scripts.evaluate_end_to_end

# Run all unit tests (29 tests)
python -m pytest tests/
```
Note: stop the API (Ctrl+C) before running the scripts, so two programs do not use the database at once.
