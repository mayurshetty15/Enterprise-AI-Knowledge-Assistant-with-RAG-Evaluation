from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable


def _source_key(item: dict[str, Any]) -> tuple[str, int]:
    document_id = item.get("document_id")
    page = item.get("page", item.get("page_number"))
    if not isinstance(document_id, str) or not isinstance(page, int):
        raise ValueError("Source results require a document_id and integer page.")
    return document_id, page


def _unique_results(results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in results:
        chunk_id = item.get("chunk_id")
        if not isinstance(chunk_id, str):
            raise ValueError("Retrieved chunks require a string chunk_id.")
        if chunk_id not in seen:
            seen.add(chunk_id)
            unique.append(item)
    return unique


def _page_ranked_chunks(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    seen_pages: set[tuple[str, int]] = set()
    for item in results:
        key = _source_key(item)
        if key not in seen_pages:
            seen_pages.add(key)
            ranked.append(item)
    return ranked


def retrieval_metrics(
    retrieved: Iterable[dict[str, Any]],
    expected_sources: Iterable[dict[str, Any]],
    relevant_chunk_ids: Iterable[str] = (),
    k_values: Iterable[int] = (1, 3, 5),
) -> dict[str, float | None]:
    """Compute ranked source-page metrics, or chunk metrics when chunk labels exist.

    Retrieved duplicate chunk IDs are removed. Without chunk labels, retrieval is
    collapsed to the first result per (document_id, page). Precision's denominator
    is min(K, unique retrieved units); recall's denominator is the number of
    expected relevant units. No-reference cases return None for retrieval metrics.
    """
    k_values = tuple(k_values)
    if any(not isinstance(k, int) or k < 1 for k in k_values):
        raise ValueError("K values must be positive integers.")
    expected = {_source_key(source) for source in expected_sources}
    exact_chunk_ids = set(relevant_chunk_ids)
    unique_chunks = _unique_results(retrieved)
    if exact_chunk_ids:
        ranked = unique_chunks
        relevant = [
            item["chunk_id"] in exact_chunk_ids
            for item in ranked
        ]
        relevant_total = len(exact_chunk_ids)
    else:
        ranked = _page_ranked_chunks(unique_chunks)
        relevant = [_source_key(item) in expected for item in ranked]
        relevant_total = len(expected)

    metrics: dict[str, float | None] = {}
    if relevant_total == 0:
        for k in k_values:
            metrics[f"precision@{k}"] = None
            metrics[f"recall@{k}"] = None
            metrics[f"hit_rate@{k}"] = None
            metrics[f"document_accuracy@{k}"] = None
            metrics[f"page_accuracy@{k}"] = None
        metrics["mrr"] = None
        return metrics

    first_relevant_rank = next(
        (index for index, is_relevant in enumerate(relevant, start=1) if is_relevant),
        None,
    )
    metrics["mrr"] = 1.0 / first_relevant_rank if first_relevant_rank else 0.0

    expected_documents = {document_id for document_id, _ in expected}
    for k in k_values:
        top = ranked[:k]
        top_relevance = relevant[:k]
        relevant_count = sum(top_relevance)
        precision_denominator = min(k, len(ranked))
        metrics[f"precision@{k}"] = (
            relevant_count / precision_denominator if precision_denominator else 0.0
        )
        metrics[f"recall@{k}"] = relevant_count / relevant_total
        metrics[f"hit_rate@{k}"] = float(relevant_count > 0)
        found_pages = {_source_key(item) for item in top}
        found_documents = {document_id for document_id, _ in found_pages}
        metrics[f"document_accuracy@{k}"] = (
            len(found_documents & expected_documents) / len(expected_documents)
            if expected_documents
            else None
        )
        metrics[f"page_accuracy@{k}"] = (
            len(found_pages & expected) / len(expected) if expected else None
        )
    return metrics


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for",
    "from", "how", "in", "is", "it", "of", "on", "or", "the", "this", "to",
    "was", "what", "when", "where", "which", "who", "why", "with",
}
_CITATION_PATTERN = re.compile(r"\[[^\]]+p\.\d+\]", re.IGNORECASE)


def _tokens(text: str, *, meaningful_only: bool = False) -> list[str]:
    tokens = _TOKEN_PATTERN.findall(text.casefold())
    if meaningful_only:
        tokens = [token for token in tokens if token not in _STOP_WORDS]
    return tokens


def token_f1(expected: str, actual: str) -> float:
    """Return multiset token F1; this is a lexical proxy, not semantic correctness."""
    expected_tokens = Counter(_tokens(expected))
    actual_tokens = Counter(_tokens(actual))
    if not expected_tokens and not actual_tokens:
        return 1.0
    if not expected_tokens or not actual_tokens:
        return 0.0
    overlap = sum((expected_tokens & actual_tokens).values())
    if not overlap:
        return 0.0
    precision = overlap / sum(actual_tokens.values())
    recall = overlap / sum(expected_tokens.values())
    return 2 * precision * recall / (precision + recall)


def lexical_faithfulness(answer: str, context: str) -> float | None:
    """Fraction of meaningful answer tokens present in retrieved text (heuristic)."""
    answer_tokens = _tokens(_CITATION_PATTERN.sub("", answer), meaningful_only=True)
    if not answer_tokens:
        return None
    context_tokens = set(_tokens(context, meaningful_only=True))
    return sum(token in context_tokens for token in answer_tokens) / len(answer_tokens)


def lexical_answer_relevance(question: str, answer: str) -> float | None:
    """Fraction of meaningful question terms echoed in the answer (heuristic)."""
    question_tokens = set(_tokens(question, meaningful_only=True))
    if not question_tokens:
        return None
    answer_tokens = set(_tokens(_CITATION_PATTERN.sub("", answer), meaningful_only=True))
    return len(question_tokens & answer_tokens) / len(question_tokens)


def citation_f1(
    expected_sources: Iterable[dict[str, Any]],
    actual_sources: Iterable[dict[str, Any]],
) -> float:
    """Exact source-document/page set F1; an empty/empty citation set scores 1."""
    expected = {_source_key(source) for source in expected_sources}
    actual = {_source_key(source) for source in actual_sources}
    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0
    overlap = len(expected & actual)
    precision = overlap / len(actual)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0
