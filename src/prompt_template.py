"""
Builds the prompt sent to the LLM. The system prompt is the main
hallucination guard: it forces the model to answer ONLY from the
provided context, and to say so explicitly when it can't.
"""

from typing import List

SYSTEM_PROMPT: str = """You are a helpful assistant that answers questions using ONLY the context provided below.

Rules you must follow:
1. Only use information found in the context. Do not use outside knowledge.
2. If the context does not contain enough information to answer the question, respond exactly with: "I don't have enough information in the provided context to answer this question."
3. Do not make up facts, names, or details that are not in the context.
4. Keep answers concise and directly relevant to the question."""


def build_user_prompt(question: str, context_chunks: List[str]) -> str:
    """
    Combine the retrieved chunks and the question into one prompt.

    Each chunk is labeled ([Context 1], [Context 2], ...) so the model can
    tell the passages apart.

    Args:
        question: The user's question.
        context_chunks: The retrieved chunk texts. May be empty.

    Returns:
        The full prompt text to send as the user message.
    """
    if not context_chunks:
        context_text = "(No relevant context was retrieved.)"
    else:
        context_text = "\n\n".join(
            f"[Context {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
        )

    return f"""Context:
{context_text}

Question: {question}

Answer based only on the context above:"""