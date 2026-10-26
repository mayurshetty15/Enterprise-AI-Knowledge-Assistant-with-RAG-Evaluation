from pathlib import Path

import pytest

from app.config import EMBEDDING_MODEL
from app.generation import generate_answer
from app.ingestion import build_document_chunks, chunk_text, stable_chunk_id
from app.retrieval import validate_question
from app.vector_store import generate_embedding


def test_chunk_text_overlap_and_length():
    text = "word " * 200
    chunks = chunk_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 200
    assert chunks[0] != chunks[1]


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_stable_chunk_id_is_repeatable():
    first = stable_chunk_id("the-file.pdf", 2, 0, "some text")
    second = stable_chunk_id("the-file.pdf", 2, 0, "some text")
    assert first == second
    assert len(first) == 64


def test_empty_question_validation_raises():
    with pytest.raises(ValueError):
        validate_question("")
    with pytest.raises(ValueError):
        validate_question("   ")


def test_generate_embedding_uses_current_ollama_embed_api(monkeypatch):
    request = {}

    def fake_request(endpoint, payload):
        request["endpoint"] = endpoint
        request["payload"] = payload
        return {"embeddings": [[0.25, -0.5, 0.75]]}

    monkeypatch.setattr("app.vector_store._ollama_request", fake_request)
    assert generate_embedding("known text") == [0.25, -0.5, 0.75]
    assert request["endpoint"] == "/api/embed"
    assert request["payload"]["input"] == "known text"


def test_generate_embedding_falls_back_for_older_ollama(monkeypatch):
    import requests

    calls = []

    def fake_request(endpoint, payload):
        calls.append((endpoint, payload))
        if endpoint == "/api/embed":
            response = requests.Response()
            response.status_code = 404
            raise requests.HTTPError("not found", response=response)
        return {"embedding": [0.1, -0.2]}

    monkeypatch.setattr("app.vector_store._ollama_request", fake_request)
    assert generate_embedding("known text") == [0.1, -0.2]
    assert calls[0][0] == "/api/embed"
    assert calls[1] == (
        "/api/embeddings",
        {"model": EMBEDDING_MODEL, "prompt": "known text"},
    )


def test_insufficient_evidence_response():
    result = generate_answer("What is the answer?", [])
    assert result["status"] == "insufficient_evidence"
    assert "could not find enough evidence" in result["answer"].lower()


def test_citation_metadata_shape():
    chunk = {
        "chunk_id": "abc",
        "source_name": "manual.pdf",
        "page_number": 3,
        "text": "This is example text for retrieval.",
        "distance": 0.12,
    }
    result = generate_answer("What is the policy?", [chunk])
    assert result["citations"][0]["source_name"] == "manual.pdf"
    assert result["citations"][0]["page_number"] == 3


def test_missing_pdf_directory_is_handled():
    missing = Path("data/does-not-exist")
    assert missing.exists() is False
    assert missing.is_dir() is False


def test_build_document_chunks_ignores_empty_pages(tmp_path):
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(b"not a real pdf")
    chunks = build_document_chunks(pdf_file)
    assert isinstance(chunks, list)
    assert chunks == []
