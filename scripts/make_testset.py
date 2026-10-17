"""
Builds a small test set for evaluation.
For 20 random chunks, asks the LLM to write one question the chunk answers.
Relevant chunks for a question = all chunks from the same story.
Saved to evaluation/testset.json and reused by both evaluation scripts.
"""

import json
import os
import random
import time

from src.llm_client import call_llm
from src.vector_store import get_collection

SAMPLE_SIZE = 20
OUTPUT_FILE = "evaluation/testset.json"

SYSTEM_PROMPT = (
    "You write test questions. Given a short story passage, write ONE clear "
    "question that can be answered using only that passage. Do not copy "
    "sentences word for word. Reply with only the question."
)


def main():
    collection = get_collection()
    data = collection.get(include=["documents", "metadatas"])
    ids, texts, metas = data["ids"], data["documents"], data["metadatas"]

    # Which chunks belong to which story
    story_to_chunks = {}
    for chunk_id, meta in zip(ids, metas):
        story_to_chunks.setdefault(meta["source_doc_id"], []).append(chunk_id)

    random.seed(11)  # same picks every time
    candidates = [i for i, t in enumerate(texts) if len(t.split()) >= 30]  # skip tiny chunks
    picks = random.sample(candidates, SAMPLE_SIZE)

    testset = []
    for n, i in enumerate(picks, start=1):
        try:
            question = call_llm(SYSTEM_PROMPT, texts[i]).strip()
        except Exception as e:
            print(f"{n}/{SAMPLE_SIZE}: skipped ({e})")
            continue

        if not question or question.startswith("Error:"):
            print(f"{n}/{SAMPLE_SIZE}: skipped ({question})")
            continue

        story_id = metas[i]["source_doc_id"]
        testset.append(
            {
                "question": question,
                "source_chunk_id": ids[i],
                "source_doc_id": story_id,
                "relevant_ids": story_to_chunks[story_id],
                "source_text": texts[i],
            }
        )
        print(f"{n}/{SAMPLE_SIZE}: {question}")
        time.sleep(2)  # stay under the free rate limit

    os.makedirs("evaluation", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(testset, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(testset)} test questions to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()