"""
End-to-end evaluation: runs test questions through the full RAG pipeline,
grades answer quality with an LLM grader, and logs latency.
"""

import json
import time

from src.embeddings import embed_chunks
from src.llm_client import call_llm
from src.rag_pipeline import answer_question
from src.vector_store import get_collection

TESTSET_FILE = "evaluation/testset.json"
REPORT_FILE = "evaluation/end_to_end_report.txt"
PAUSE_SECONDS = 5  # stay under the free API rate limit

OFF_TOPIC_QUESTIONS = [
    "What is the capital of France?",
    "How do I install Python on Windows?",
    "Who won the 2022 football world cup?",
]

JUDGE_SYSTEM_PROMPT = (
    "You are a strict grader. You will get a passage, a question and an answer. "
    "Reply with exactly one word: CORRECT if the answer is right according to the "
    "passage, or INCORRECT if it is wrong or incomplete."
)


def is_refusal(answer: str) -> bool:
    return "enough information" in answer.lower()


def judge(passage, question, answer):
    """Returns True (correct), False (incorrect) or None (grader failed)."""
    prompt = f"Passage:\n{passage}\n\nQuestion: {question}\n\nAnswer: {answer}\n\nVerdict:"
    verdict = call_llm(JUDGE_SYSTEM_PROMPT, prompt).strip().upper()
    if verdict.startswith("INCORRECT"):  # check this first: "CORRECT" is inside "INCORRECT"
        return False
    if verdict.startswith("CORRECT"):
        return True
    return None


def percent(part, whole):
    return f"{part / whole * 100:.0f}%" if whole else "n/a"


def average(values):
    return sum(values) / len(values) if values else 0.0


def main():
    with open(TESTSET_FILE, "r", encoding="utf-8") as f:
        testset = json.load(f)

    if not testset:
        print("Test set is empty. Run python -m scripts.make_testset first.")
        return

    # Warm-up so the first question is not slowed by one-time startup
    get_collection()
    embed_chunks(["warm up"])

    # ---- In-domain questions ----
    rows = []
    for n, item in enumerate(testset, start=1):
        result = answer_question(item["question"])
        answer = result["answer"]

        correct = None
        if answer.startswith("Error:"):
            status = "error"
        elif is_refusal(answer):
            status = "refused"
            correct = False
        else:
            status = "answered"
            correct = judge(item["source_text"], item["question"], answer)

        rows.append(
            {
                "question": item["question"],
                "answer": answer,
                "status": status,
                "correct": correct,
                "source_retrieved": item["source_text"] in result["retrieved_chunks"],
                "retrieval_s": result["retrieval_time_s"],
                "generation_s": result["generation_time_s"],
                "total_s": result["total_time_s"],
            }
        )
        print(f"\n>>> {n}/{len(testset)}: {status} | correct={correct}")
        time.sleep(PAUSE_SECONDS)

    # ---- Off-topic questions ----
    off_topic_refused = 0
    for question in OFF_TOPIC_QUESTIONS:
        result = answer_question(question)
        if result["out_of_domain"] or is_refusal(result["answer"]):
            off_topic_refused += 1
        time.sleep(PAUSE_SECONDS)

    # ---- Numbers ----
    total = len(rows)
    errors = sum(1 for r in rows if r["status"] == "error")
    refused = sum(1 for r in rows if r["status"] == "refused")
    graded = [r for r in rows if r["correct"] is not None]
    grader_failures = total - len(graded) - errors

    correct_count = sum(1 for r in graded if r["correct"])
    with_source = [r for r in graded if r["source_retrieved"]]
    without_source = [r for r in graded if not r["source_retrieved"]]
    correct_with = sum(1 for r in with_source if r["correct"])
    correct_without = sum(1 for r in without_source if r["correct"])

    gen_times = [r["generation_s"] for r in rows if r["generation_s"] > 0]

    lines = [
        "END-TO-END EVALUATION",
        "=" * 50,
        f"In-domain questions: {total}",
        f"Graded: {len(graded)} (API errors: {errors}, grader failures: {grader_failures})",
        f"Correct answers: {correct_count}/{len(graded)} ({percent(correct_count, len(graded))})",
        f"Refused (said 'not enough information'): {refused}/{total}",
        f"Accuracy when the source chunk WAS retrieved: {correct_with}/{len(with_source)} ({percent(correct_with, len(with_source))})",
        f"Accuracy when the source chunk was NOT retrieved: {correct_without}/{len(without_source)} ({percent(correct_without, len(without_source))})",
        f"Off-topic questions refused: {off_topic_refused}/{len(OFF_TOPIC_QUESTIONS)}",
        "",
        f"Average retrieval time: {average([r['retrieval_s'] for r in rows]):.3f} s",
        f"Average generation time: {average(gen_times):.3f} s",
        f"Average total time: {average([r['total_s'] for r in rows]):.3f} s",
        f"Slowest total time: {max(r['total_s'] for r in rows):.3f} s",
        "",
        "PER QUESTION",
        "-" * 50,
    ]
    for n, r in enumerate(rows, start=1):
        lines.append(
            f"{n:>2} | {r['status']:<8} | correct={str(r['correct']):<5} | "
            f"source_retrieved={str(r['source_retrieved']):<5} | "
            f"{r['total_s']:.2f}s | {r['question'][:60]}"
        )

    report = "\n".join(lines)
    print("\n" + report)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to {REPORT_FILE}")


if __name__ == "__main__":
    main()