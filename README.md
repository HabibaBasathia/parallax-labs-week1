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