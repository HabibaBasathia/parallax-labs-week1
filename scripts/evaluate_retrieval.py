"""
Retrieval evaluation: Precision@K and Recall@K on the test set.
Relevant chunks for a question = all chunks from the same story.
"""

import json
import time

from src.embeddings import get_model
from src.vector_store import get_collection

TESTSET_FILE = "evaluation/testset.json"
REPORT_FILE = "evaluation/retrieval_report.txt"
K_VALUES = [1, 3, 5]


def main():
    with open(TESTSET_FILE, "r", encoding="utf-8") as f:
        testset = json.load(f)

    if not testset:
        print("Test set is empty. Run python -m scripts.make_testset first.")
        return

    model = get_model()
    collection = get_collection()
    max_k = max(K_VALUES)

    model.encode(["warm up"])  # warm-up so timing is fair

    precision_sums = {k: 0.0 for k in K_VALUES}
    recall_sums = {k: 0.0 for k in K_VALUES}
    hit_sums = {k: 0 for k in K_VALUES}
    latencies = []
    misses = []

    for item in testset:
        relevant = set(item["relevant_ids"])

        start = time.perf_counter()
        embedding = model.encode([item["question"]])[0].tolist()
        results = collection.query(query_embeddings=[embedding], n_results=max_k)
        latencies.append(time.perf_counter() - start)

        retrieved = results["ids"][0]

        for k in K_VALUES:
            top = retrieved[:k]
            found = sum(1 for chunk_id in top if chunk_id in relevant)
            precision_sums[k] += found / k
            recall_sums[k] += found / len(relevant)
            if item["source_chunk_id"] in top:
                hit_sums[k] += 1

        if item["source_chunk_id"] not in retrieved:
            misses.append(item)

    total = len(testset)
    avg_relevant = sum(len(i["relevant_ids"]) for i in testset) / total
    avg_latency_ms = sum(latencies) / total * 1000

    lines = []
    lines.append("RETRIEVAL EVALUATION")
    lines.append("=" * 50)
    lines.append(f"Test questions: {total}")
    lines.append(f"Average relevant chunks per question: {avg_relevant:.1f}")
    lines.append(f"Average search time: {avg_latency_ms:.1f} ms")
    lines.append("")
    lines.append("K | Precision@K | Recall@K | Source chunk found")
    for k in K_VALUES:
        lines.append(
            f"{k} | {precision_sums[k] / total:.2f}        | "
            f"{recall_sums[k] / total:.2f}     | {hit_sums[k]}/{total}"
        )
    lines.append("")
    lines.append(f"Questions where the source chunk was NOT in the top {max_k}: {len(misses)}")
    for item in misses:
        lines.append("-" * 50)
        lines.append(f"Q: {item['question']}")
        lines.append(f"Source text: {item['source_text'][:200]}")

    report = "\n".join(lines)
    print(report)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to {REPORT_FILE}")


if __name__ == "__main__":
    main()