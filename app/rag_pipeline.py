from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import PDF_DIR, ensure_project_paths
from app.generation import generate_answer
from app.ingestion import build_document_chunks, list_pdf_files
from app.retrieval import retrieve_relevant_chunks, validate_question
from app.vector_store import add_document_chunks, get_document_sources


class RAGPipeline:
    """Small orchestration layer for local PDF ingestion, retrieval, and answer generation."""

    def __init__(self, pdf_dir: str | Path = PDF_DIR):
        self.pdf_dir = Path(pdf_dir)
        ensure_project_paths()

    def ingest_documents(self) -> list[str]:
        """Index all PDFs in the configured directory."""
        if not self.pdf_dir.exists():
            raise FileNotFoundError(f"PDF directory does not exist: {self.pdf_dir}")

        pdf_files = list_pdf_files(self.pdf_dir)
        indexed_files: list[str] = []
        for file_path in pdf_files:
            chunks = build_document_chunks(file_path)
            if not chunks:
                continue
            add_document_chunks(chunks)
            indexed_files.append(file_path.name)
        return indexed_files

    def ask(self, question: str) -> dict[str, Any]:
        """Answer a user question using the local RAG pipeline."""
        cleaned_question = validate_question(question)
        retrieved_chunks = retrieve_relevant_chunks(cleaned_question, n_results=5)
        if not retrieved_chunks:
            return {
                "question": cleaned_question,
                "answer": "I could not find enough evidence in the indexed documents to answer this question reliably.",
                "citations": [],
                "retrieved_chunks": [],
                "status": "insufficient_evidence",
            }

        result = generate_answer(cleaned_question, retrieved_chunks)
        return {
            "question": cleaned_question,
            "answer": result["answer"],
            "citations": result.get("citations", []),
            "retrieved_chunks": retrieved_chunks,
            "status": result.get("status", "answer_ready"),
        }

    def list_sources(self) -> list[str]:
        return get_document_sources()


def build_pipeline() -> RAGPipeline:
    return RAGPipeline(pdf_dir=PDF_DIR)
