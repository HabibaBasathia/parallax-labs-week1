"""
Runs a few test questions through the full RAG pipeline, including
one intentionally off-topic question to confirm the hallucination
guard works.
"""

from src.rag_pipeline import answer_question

TEST_QUESTIONS = [
    "What happens when the two friends go to the park?",
    "What lesson does the little boy learn?",
    # Intentionally off-topic / out-of-domain question:
    "What is the capital of France?",
]


def main():
    for question in TEST_QUESTIONS:
        result = answer_question(question)

        print("=" * 60)
        print(f"Q: {question}")
        print(f"A: {result['answer']}")
        print(f"Out of domain: {result['out_of_domain']}")
        print(
            f"Latency -> retrieval: {result['retrieval_time_s']}s | "
            f"generation: {result['generation_time_s']}s | "
            f"total: {result['total_time_s']}s"
        )


if __name__ == "__main__":
    main()