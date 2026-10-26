from __future__ import annotations

from app.vector_store import query_chunks


def validate_question(question: str) -> str:
    """Reject empty or whitespace-only questions."""
    if question is None:
        raise ValueError("Question is required")
    cleaned = str(question).strip()
    if not cleaned:
        raise ValueError("Question cannot be empty")
    return cleaned


def retrieve_relevant_chunks(question: str, n_results: int = 5) -> list[dict]:
    """Validate the question and retrieve the most relevant chunks."""
    cleaned_question = validate_question(question)
    return query_chunks(cleaned_question, n_results=n_results)
