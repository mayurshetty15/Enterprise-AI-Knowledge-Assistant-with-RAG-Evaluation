import json

from fastapi.testclient import TestClient
import pytest

from app.main import create_app
from app.services.document_service import DocumentService
from app.services.evaluation_reports import EvaluationReportService


def _report(run_id):
    return {
        "metadata": {
            "run_id": run_id,
            "dataset_version": "synthetic-v1",
            "dataset_sha256": "hash",
            "generated_at": "2026-01-01T00:00:00+00:00",
            "case_count": 1,
            "embedding_model": "nomic-embed-text",
            "generation_model": "gemma2:2b",
            "configuration": {},
            "local_judge_is_subjective": False,
            "cost": "not_applicable_local_ollama",
            "private_answer": "must not be exposed",
        },
        "aggregates": {"retrieval": {"hit_rate@1": {"mean": 1, "n": 1}}},
        "cases": [
            {
                "case_id": "synthetic-case",
                "category": "factual_lookup",
                "retrieval_metrics": {},
                "answer_quality": {},
                "performance": {},
                "expected": {"pages": [2], "answer": "private"},
                "actual": {"pages": [2], "answer": "private"},
                "failure_kinds": [],
                "private_content": "must not be exposed",
            }
        ],
        "failures": [],
    }


def test_evaluation_reports_api_is_read_only_and_redacts_content(tmp_path):
    run_id = "20260101T000000000000Z_abc123_abc123_abc123"
    run_directory = tmp_path / run_id
    run_directory.mkdir()
    (run_directory / "report.json").write_text(json.dumps(_report(run_id)), encoding="utf-8")

    app = create_app(
        document_service=DocumentService(tmp_path / "documents.sqlite3"),
        health_check=lambda: {"status": "ok", "services": {"api": "ok"}},
    )
    app.state.evaluation_report_service = EvaluationReportService(tmp_path)
    client = TestClient(app)

    listed = client.get("/api/evaluation/runs")
    assert listed.status_code == 200
    assert listed.json()["runs"][0]["metadata"]["run_id"] == run_id
    detail = client.get(f"/api/evaluation/runs/{run_id}")
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["run"]["cases"][0]["actual"]["pages"] == [2]
    assert "private_answer" not in payload["run"]["metadata"]
    assert "private_content" not in payload["run"]["cases"][0]

    traversal = client.get("/api/evaluation/runs/%2E%2E")
    missing = client.get("/api/evaluation/runs/no-such-run")
    assert traversal.status_code == 404
    assert missing.status_code == 404


def test_evaluation_report_service_rejects_symlink_paths(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    run_id = "20260101T000000000000Z_abc123_abc123"
    (outside / "report.json").write_text(json.dumps(_report(run_id)), encoding="utf-8")
    try:
        (reports_dir / run_id).symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Directory symlinks are unavailable in this Windows environment.")

    service = EvaluationReportService(reports_dir)
    assert service.list_runs() == []
    assert service.get_run(run_id) is None
