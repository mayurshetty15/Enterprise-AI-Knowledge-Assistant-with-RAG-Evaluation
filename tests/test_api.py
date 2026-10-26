from __future__ import annotations

import fitz
import pytest
from fastapi.testclient import TestClient

from app.config import EMBEDDING_MODEL, GENERATION_MODEL
from app.main import create_app
from app.services.document_service import DocumentService, check_services
from app.services.rag_service import RAGService


class FakeCollection:
    def __init__(self):
        self.chunks = {}

    def get(self, where=None, include=None):
        values = list(self.chunks.values())
        if where:
            field, condition = next(iter(where.items()))
            if isinstance(condition, dict):
                ids = set(condition["$in"])
                values = [item for item in values if item.get(field) in ids]
            else:
                values = [item for item in values if item.get(field) == condition]
        return {"metadatas": values, "ids": list(self.chunks)}

    def update(self, ids, metadatas):
        for chunk_id, metadata in zip(ids, metadatas):
            self.chunks[chunk_id] = metadata

    def count(self):
        return len(self.chunks)


@pytest.fixture
def client(tmp_path, monkeypatch):
    collection = FakeCollection()
    monkeypatch.setattr(
        "app.services.document_service.get_collection", lambda: collection
    )

    def add_chunks(chunks):
        for chunk in chunks:
            collection.chunks[chunk["chunk_id"]] = {
                "source_name": chunk["source_name"],
                "document_id": chunk["document_id"],
                "page_number": chunk["page_number"],
            }
        return len(chunks)

    def delete_chunks(source_name=None, document_id=None):
        matching = [
            key
            for key, metadata in collection.chunks.items()
            if (document_id and metadata.get("document_id") == document_id)
            or (source_name and metadata.get("source_name") == source_name)
        ]
        for key in matching:
            del collection.chunks[key]
        return len(matching)

    monkeypatch.setattr("app.services.document_service.add_document_chunks", add_chunks)
    monkeypatch.setattr(
        "app.services.document_service.delete_document_chunks", delete_chunks
    )

    documents = DocumentService(tmp_path / "documents.sqlite3")

    class FakeRAGService:
        result = {
            "answer": "The report identifies three findings.",
            "sources": [
                {
                    "document_id": "doc-1",
                    "filename": "report.pdf",
                    "page": 2,
                    "snippet": "Three findings are summarized here.",
                }
            ],
            "retrieval_count": 1,
            "status": "answer_ready",
        }

        def ask(self, question, document_ids=None):
            if question == "fail":
                raise RuntimeError("mocked local service failure")
            if question == "no evidence":
                return {
                    "answer": "I could not find enough evidence.",
                    "sources": [],
                    "retrieval_count": 0,
                    "status": "insufficient_evidence",
                }
            return self.result

    app = create_app(
        document_service=documents,
        rag_service=FakeRAGService(),
        health_check=lambda: {
            "status": "degraded",
            "services": {
                "api": "ok",
                "ollama": {"status": "unavailable", "reachable": False},
                "chat_model": {"status": "unavailable", "name": "gemma2:2b"},
                "embedding_model": {"status": "unavailable", "name": "nomic-embed-text"},
                "vector_database": {"status": "ok"},
            },
        },
    )
    app.state.test_collection = collection
    return TestClient(app)


def make_pdf() -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "The report identifies three important findings.")
    contents = document.tobytes()
    document.close()
    return contents


def test_health_endpoint_reports_component_statuses(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["services"]["ollama"]["reachable"] is False
    assert payload["services"]["vector_database"]["status"] == "ok"


def test_health_check_matches_ollama_default_latest_tag(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "models": [
                    {"name": f"{EMBEDDING_MODEL}:latest"},
                    {"name": GENERATION_MODEL},
                ]
            }

    monkeypatch.setattr("requests.get", lambda url, timeout: FakeResponse())
    monkeypatch.setattr(
        "app.services.document_service.get_collection", lambda: FakeCollection()
    )
    health = check_services()
    assert health["status"] == "ok"
    assert health["services"]["embedding_model"]["status"] == "available"
    assert health["services"]["chat_model"]["status"] == "available"


def test_upload_requires_pdf_file(client):
    missing = client.post("/api/documents/upload")
    invalid = client.post(
        "/api/documents/upload",
        files={"file": ("notes.txt", b"not a pdf", "text/plain")},
    )
    assert missing.status_code == 422
    assert invalid.status_code == 415


def test_upload_indexes_valid_pdf_and_lists_then_deletes_it(client):
    upload = client.post(
        "/api/documents/upload",
        files={"file": ("report.pdf", make_pdf(), "application/pdf")},
    )
    assert upload.status_code == 201
    uploaded = upload.json()
    assert uploaded["filename"] == "report.pdf"
    assert uploaded["page_count"] == 1
    assert uploaded["chunk_count"] == 1
    assert uploaded["status"] == "indexed"

    listing = client.get("/api/documents")
    assert listing.status_code == 200
    assert listing.json()["documents"][0]["document_id"] == uploaded["document_id"]

    deleted = client.delete(f"/api/documents/{uploaded['document_id']}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted_chunks"] == 1
    assert client.get("/api/documents").json()["documents"] == []
    assert client.delete(f"/api/documents/{uploaded['document_id']}").status_code == 404


def test_upload_rejects_corrupt_pdf(client):
    response = client.post(
        "/api/documents/upload",
        files={"file": ("broken.pdf", b"not a PDF", "application/pdf")},
    )
    assert response.status_code == 422
    assert "readable PDF" in response.json()["detail"]


def test_upload_respects_configured_size_limit(client, monkeypatch):
    monkeypatch.setattr("app.api.routes.MAX_UPLOAD_SIZE_BYTES", 3)
    response = client.post(
        "/api/documents/upload",
        files={"file": ("large.pdf", b"1234", "application/pdf")},
    )
    assert response.status_code == 413


def test_duplicate_upload_and_legacy_delete_do_not_remove_uploaded_chunks(client):
    contents = make_pdf()
    upload = client.post(
        "/api/documents/upload",
        files={"file": ("report.pdf", contents, "application/pdf")},
    )
    duplicate = client.post(
        "/api/documents/upload",
        files={"file": ("report.pdf", contents, "application/pdf")},
    )
    assert duplicate.status_code == 200
    assert duplicate.json()["already_indexed"] is True

    collection = client.app.state.test_collection
    collection.chunks["phase-one-chunk"] = {
        "source_name": "report.pdf",
        "page_number": 7,
    }
    documents = client.get("/api/documents").json()["documents"]
    legacy = next(item for item in documents if item["document_id"].startswith("legacy:"))
    assert legacy["chunk_count"] == 1
    assert client.app.state.document_service.resolve_document_ids(
        [legacy["document_id"]]
    ) == ([legacy["document_id"]], [])

    deleted = client.delete(f"/api/documents/{legacy['document_id']}")
    assert deleted.status_code == 200
    assert "phase-one-chunk" not in collection.chunks
    assert any(
        metadata.get("document_id") == upload.json()["document_id"]
        for metadata in collection.chunks.values()
    )


def test_upload_indexing_failure_is_reported_and_marked(client, monkeypatch):
    def fail_indexing(chunks):
        raise RuntimeError("mocked embedding failure")

    monkeypatch.setattr("app.services.document_service.add_document_chunks", fail_indexing)
    contents = make_pdf()
    response = client.post(
        "/api/documents/upload",
        files={"file": ("report.pdf", contents, "application/pdf")},
    )
    assert response.status_code == 503
    documents = client.get("/api/documents").json()["documents"]
    assert documents[0]["status"] == "failed"

    def retry_indexing(chunks):
        for chunk in chunks:
            client.app.state.test_collection.chunks[chunk["chunk_id"]] = {
                "source_name": chunk["source_name"],
                "document_id": chunk["document_id"],
                "page_number": chunk["page_number"],
            }
        return len(chunks)

    monkeypatch.setattr("app.services.document_service.add_document_chunks", retry_indexing)
    retried = client.post(
        "/api/documents/upload",
        files={"file": ("report.pdf", contents, "application/pdf")},
    )
    assert retried.status_code == 201
    assert retried.json()["status"] == "indexed"


def test_selected_documents_are_passed_to_phase_one_retrieval(client, monkeypatch):
    upload = client.post(
        "/api/documents/upload",
        files={"file": ("report.pdf", make_pdf(), "application/pdf")},
    )
    document_id = upload.json()["document_id"]
    observed = {}

    def retrieve(question, **filters):
        observed.update(filters)
        return []

    monkeypatch.setattr("app.services.rag_service.retrieve_relevant_chunks", retrieve)
    monkeypatch.setattr(
        "app.services.rag_service.generate_answer",
        lambda question, chunks: {
            "answer": "I could not find enough evidence.",
            "status": "insufficient_evidence",
        },
    )
    service = RAGService(client.app.state.document_service)
    result = service.ask("What did the report say?", [document_id])
    assert result["status"] == "insufficient_evidence"
    assert observed["document_ids"] == [document_id]
    assert observed["source_names"] is None


def test_question_validation_and_grounded_response(client):
    empty = client.post("/api/ask", json={"question": "   "})
    answered = client.post("/api/ask", json={"question": "What did the report find?"})
    assert empty.status_code == 422
    assert answered.status_code == 200
    assert answered.json()["sources"][0]["page"] == 2
    assert answered.json()["retrieval_count"] == 1


def test_empty_retrieval_and_service_failure_are_reported(client):
    no_evidence = client.post("/api/ask", json={"question": "no evidence"})
    unavailable = client.post("/api/ask", json={"question": "fail"})
    assert no_evidence.json()["sources"] == []
    assert no_evidence.json()["status"] == "insufficient_evidence"
    assert unavailable.status_code == 503
