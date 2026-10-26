from __future__ import annotations

import hashlib
from typing import Any

from app.config import MAX_RETRIEVAL_RESULTS
from app.generation import generate_answer
from app.retrieval import retrieve_relevant_chunks, validate_question
from app.services.document_service import DocumentService


class RAGService:
    """Adapt the Phase 1 retrieval and answer generation pipeline for the API."""

    def __init__(self, documents: DocumentService):
        self.documents = documents

    def ask(self, question: str, document_ids: list[str] | None = None) -> dict[str, Any]:
        cleaned_question = validate_question(question)
        selected_ids, source_names = self.documents.resolve_document_ids(document_ids or [])
        chunks = retrieve_relevant_chunks(
            cleaned_question,
            n_results=MAX_RETRIEVAL_RESULTS,
            document_ids=selected_ids or None,
            source_names=source_names or None,
        )
        result = generate_answer(cleaned_question, chunks)
        sources = []
        seen: set[tuple[str, str, int]] = set()
        for chunk in chunks:
            filename = chunk.get("source_name", "unknown")
            page = int(chunk.get("page_number", 1))
            document_id = chunk.get("document_id") or (
                "legacy:" + hashlib.sha256(filename.encode("utf-8")).hexdigest()[:24]
            )
            key = (document_id, filename, page)
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                {
                    "document_id": document_id,
                    "filename": filename,
                    "page": page,
                    "snippet": chunk.get("text", "")[:600],
                }
            )
        return {
            "answer": result["answer"],
            "sources": sources,
            "retrieval_count": len(chunks),
            "status": result.get("status", "answer_ready"),
        }
