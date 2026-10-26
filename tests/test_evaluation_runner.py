import csv
import json
from pathlib import Path

import pytest

from evaluation.runner import run_evaluation


def _dataset(path: Path):
    path.write_text(
        json.dumps(
            {
                "dataset_version": "test-v1",
                "description": "Private test fixture.",
                "corpus": [
                    {
                        "document_id": "doc-a",
                        "filename": "safe-id.pdf",
                        "pages": [{"page": 2, "text": "The answer is CEDAR-482."}],
                    }
                ],
                "cases": [
                    {
                        "case_id": "case-a",
                        "question": "PRIVATE QUESTION SECRET?",
                        "reference_answer": "CEDAR-482",
                        "expected_sources": [{"document_id": "doc-a", "page": 2}],
                        "relevant_chunk_ids": [],
                        "category": "factual_lookup",
                        "answerable": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_runner_uses_isolated_collection_and_writes_redacted_unique_reports(
    tmp_path, monkeypatch
):
    from app import vector_store
    from evaluation import runner

    dataset_path = tmp_path / "dataset.json"
    _dataset(dataset_path)
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(runner, "generate_embedding", lambda text: [1.0, 0.0])
    monkeypatch.setattr(vector_store, "generate_embedding", lambda text: [1.0, 0.0])

    def answer(question, chunks):
        assert chunks[0]["document_id"] == "doc-a"
        return {
            "answer": "CEDAR-482",
            "status": "answer_ready",
            "citations": [{"source_name": "safe-id.pdf", "page_number": 2}],
        }

    monkeypatch.setattr(runner, "generate_answer", answer)
    result = run_evaluation(dataset_path, reports_dir)
    run_directory = reports_dir / result["metadata"]["run_id"]
    assert (run_directory / "report.json").exists()
    assert (run_directory / "cases.csv").exists()
    assert (run_directory / "report.md").exists()
    assert result["aggregates"]["retrieval"]["page_accuracy@1"]["mean"] == 1
    assert result["aggregates"]["answer_quality"]["answer_correctness"]["mean"] == 1

    report_text = (run_directory / "report.json").read_text(encoding="utf-8")
    markdown_text = (run_directory / "report.md").read_text(encoding="utf-8")
    csv_text = (run_directory / "cases.csv").read_text(encoding="utf-8")
    for private_text in ("PRIVATE QUESTION SECRET", "CEDAR-482"):
        assert private_text not in report_text
        assert private_text not in markdown_text
        assert private_text not in csv_text
    with (run_directory / "cases.csv").open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert rows[0]["case_id"] == "case-a"
    with pytest.raises(FileExistsError):
        (reports_dir / result["metadata"]["run_id"]).mkdir(exist_ok=False)


def test_runner_counts_errors_without_writing_answer_content(tmp_path, monkeypatch):
    from app import vector_store
    from evaluation import runner

    dataset_path = tmp_path / "dataset.json"
    _dataset(dataset_path)
    monkeypatch.setattr(runner, "generate_embedding", lambda text: [1.0, 0.0])
    monkeypatch.setattr(vector_store, "generate_embedding", lambda text: [1.0, 0.0])

    def retrieval_error(*args, **kwargs):
        import requests

        raise requests.Timeout("private timeout details")

    monkeypatch.setattr(runner, "retrieve_relevant_chunks", retrieval_error)
    result = run_evaluation(dataset_path, tmp_path / "reports")
    assert result["aggregates"]["failures"]["case_errors"] == 1
    assert result["aggregates"]["failures"]["timeouts"] == 1
    assert result["cases"][0]["failure_kinds"] == ["retrieval_timeout"]
    report_path = tmp_path / "reports" / result["metadata"]["run_id"] / "report.json"
    assert "private timeout details" not in report_path.read_text(encoding="utf-8")


def test_runner_scores_correct_abstention_without_reference_answer(tmp_path, monkeypatch):
    from app import vector_store
    from evaluation import runner

    dataset_path = tmp_path / "dataset.json"
    _dataset(dataset_path)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    case = dataset["cases"][0]
    case["reference_answer"] = None
    case["expected_sources"] = []
    case["answerable"] = False
    case["category"] = "insufficient_evidence"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    monkeypatch.setattr(runner, "generate_embedding", lambda text: [1.0, 0.0])
    monkeypatch.setattr(vector_store, "generate_embedding", lambda text: [1.0, 0.0])
    monkeypatch.setattr(runner, "retrieve_relevant_chunks", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        runner,
        "generate_answer",
        lambda question, chunks: {
            "answer": "Insufficient evidence.",
            "status": "insufficient_evidence",
            "citations": [],
        },
    )

    result = run_evaluation(dataset_path, tmp_path / "reports")
    quality = result["cases"][0]["answer_quality"]
    assert quality["answer_correctness"] is None
    assert quality["abstention_correct"] == 1
    assert quality["citation_f1"] == 1
    assert "abstention_mismatch" not in result["cases"][0]["failure_kinds"]


def test_runner_aggregates_optional_local_judge_scores(tmp_path, monkeypatch):
    from app import vector_store
    from evaluation import judge, runner

    dataset_path = tmp_path / "dataset.json"
    _dataset(dataset_path)
    monkeypatch.setattr(runner, "generate_embedding", lambda text: [1.0, 0.0])
    monkeypatch.setattr(vector_store, "generate_embedding", lambda text: [1.0, 0.0])
    monkeypatch.setattr(
        runner,
        "retrieve_relevant_chunks",
        lambda *args, **kwargs: [
            {
                "chunk_id": "known-chunk",
                "document_id": "doc-a",
                "source_name": "safe-id.pdf",
                "page_number": 2,
                "text": "The answer is CEDAR-482.",
                "distance": 0.1,
            }
        ],
    )
    monkeypatch.setattr(
        runner,
        "generate_answer",
        lambda question, chunks: {
            "answer": "CEDAR-482",
            "status": "answer_ready",
            "citations": [{"source_name": "safe-id.pdf", "page_number": 2}],
        },
    )
    monkeypatch.setattr(
        judge,
        "judge_answer",
        lambda *args: {
            "correctness": 0.8,
            "faithfulness": 0.6,
            "relevance": 1.0,
        },
    )

    result = run_evaluation(
        dataset_path, tmp_path / "reports", llm_judge=True
    )
    assert result["aggregates"]["answer_quality"]["local_judge_correctness"] == {
        "mean": 0.8,
        "n": 1,
    }
    assert result["aggregates"]["answer_quality"]["local_judge_faithfulness"] == {
        "mean": 0.6,
        "n": 1,
    }
