import pytest

from evaluation.metrics import (
    citation_f1,
    lexical_answer_relevance,
    lexical_faithfulness,
    retrieval_metrics,
    token_f1,
)
from evaluation.schemas import EvaluationDataset


def _chunk(chunk_id, document_id, page):
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "page": page,
    }


def test_page_metrics_rank_unique_pages_and_ignore_duplicate_chunk_ids():
    results = [
        _chunk("a2", "doc-a", 2),
        _chunk("a2", "doc-a", 2),
        _chunk("b1", "doc-b", 1),
        _chunk("a3", "doc-a", 3),
    ]
    expected = [
        {"document_id": "doc-a", "page": 2},
        {"document_id": "doc-a", "page": 3},
    ]
    metrics = retrieval_metrics(results, expected, k_values=(1, 3))
    assert metrics["precision@1"] == 1
    assert metrics["recall@1"] == 0.5
    assert metrics["mrr"] == 1
    assert metrics["hit_rate@3"] == 1
    assert metrics["document_accuracy@3"] == 1
    assert metrics["page_accuracy@3"] == 1


def test_empty_retrieval_has_zero_scores_when_relevance_is_known():
    metrics = retrieval_metrics(
        [],
        [{"document_id": "doc-a", "page": 2}],
        k_values=(1,),
    )
    assert metrics["precision@1"] == 0
    assert metrics["recall@1"] == 0
    assert metrics["hit_rate@1"] == 0
    assert metrics["mrr"] == 0


def test_no_relevant_sources_is_not_scored_as_a_retrieval_miss():
    metrics = retrieval_metrics([_chunk("a1", "doc-a", 1)], [], k_values=(1,))
    assert metrics["precision@1"] is None
    assert metrics["recall@1"] is None
    assert metrics["hit_rate@1"] is None
    assert metrics["mrr"] is None
    assert metrics["page_accuracy@1"] is None


def test_exact_chunk_relevance_uses_chunk_labels_and_deduplicates():
    metrics = retrieval_metrics(
        [_chunk("relevant", "doc-a", 2), _chunk("relevant", "doc-a", 2)],
        [],
        relevant_chunk_ids=["relevant"],
        k_values=(1,),
    )
    assert metrics["precision@1"] == 1
    assert metrics["recall@1"] == 1
    assert metrics["hit_rate@1"] == 1


def test_invalid_k_and_missing_retrieval_fields_are_rejected():
    with pytest.raises(ValueError, match="positive"):
        retrieval_metrics([], [], k_values=(0,))
    with pytest.raises(ValueError, match="chunk_id"):
        retrieval_metrics([{"document_id": "doc", "page": 1}], [])


def test_answer_and_citation_metric_edge_cases():
    assert token_f1("CEDAR-482", "The code is CEDAR 482.") > 0.5
    assert token_f1("reference", "") == 0
    assert lexical_faithfulness("CEDAR-482 is the code.", "CEDAR-482 is the code.") == 1
    assert lexical_answer_relevance("What is the access code?", "The access code is CEDAR-482.") == 1
    assert citation_f1([], []) == 1
    assert citation_f1(
        [{"document_id": "doc-a", "page": 2}],
        [{"document_id": "doc-a", "page": 3}],
    ) == 0
    assert retrieval_metrics(
        [_chunk("a2", "doc-a", 2)],
        [{"document_id": "doc-a", "page": 3}],
        k_values=(1,),
    )["page_accuracy@1"] == 0


def test_dataset_validation_rejects_missing_or_malformed_fields():
    with pytest.raises(ValueError, match="Could not read evaluation dataset"):
        from evaluation.runner import load_dataset

        load_dataset("missing-evaluation-dataset.json")

    with pytest.raises(Exception):
        EvaluationDataset.model_validate(
            {
                "dataset_version": "bad",
                "description": "missing case fields",
                "corpus": [{"document_id": "d", "filename": "../private.pdf", "pages": [{"page": 1, "text": "text"}]}],
                "cases": [{"case_id": "c", "question": "q", "answerable": True}],
            }
        )

    with pytest.raises(ValueError, match="must not declare a reference_answer"):
        EvaluationDataset.model_validate(
            {
                "dataset_version": "bad",
                "description": "unanswerable case with a reference",
                "corpus": [{"document_id": "d", "filename": "doc.pdf", "pages": [{"page": 1, "text": "text"}]}],
                "cases": [{
                    "case_id": "c",
                    "question": "q",
                    "reference_answer": "unsupported",
                    "expected_sources": [],
                    "answerable": False,
                    "category": "insufficient_evidence",
                }],
            }
        )
