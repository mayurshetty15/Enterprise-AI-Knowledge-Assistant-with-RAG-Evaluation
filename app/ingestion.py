from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import fitz

from app.config import CHUNK_OVERLAP, MAX_CHUNK_CHARS, PDF_DIR


def normalize_text(text: str) -> str:
    """Collapse repeated whitespace while preserving useful sentence boundaries."""
    if not text:
        return ""
    cleaned = " ".join(text.replace("\xa0", " ").split())
    return cleaned.strip()


def extract_pdf_pages(file_path: Path) -> list[dict[str, Any]]:
    """Return page-level text entries with page numbers and original filename."""
    pages: list[dict[str, Any]] = []
    try:
        with fitz.open(str(file_path)) as document:
            for page_number in range(1, document.page_count + 1):
                page = document[page_number - 1]
                raw_text = page.get_text("text")
                text = normalize_text(raw_text)
                if not text:
                    continue
                pages.append(
                    {
                        "source_name": file_path.name,
                        "page_number": page_number,
                        "text": text,
                    }
                )
    except (RuntimeError, ValueError, fitz.FileDataError):
        # PyMuPDF raises a few different exceptions for unreadable PDFs.
        return []
    return pages


def chunk_text(text: str, chunk_size: int = MAX_CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into chunks with overlap while keeping output deterministic."""
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 2)

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]
        if len(chunk.strip()) == 0:
            break
        cleaned_chunk = chunk.strip()
        if not chunks or cleaned_chunk != chunks[-1]:
            chunks.append(cleaned_chunk)
        if end >= len(text):
            break
        next_start = max(start + chunk_size - overlap, 0 if start == 0 else start + 1)
        next_end = min(len(text), next_start + chunk_size)
        if chunks and text[next_start:next_end].strip() == chunks[-1]:
            # Avoid identical adjacent windows when the input repeats periodically.
            next_start += 1
        start = next_start
    return chunks


def stable_chunk_id(source_name: str, page_number: int, chunk_index: int, chunk_text: str) -> str:
    """Create a deterministic, stable identifier for a chunk."""
    digest = hashlib.sha256()
    digest.update(f"{source_name}|{page_number}|{chunk_index}|{chunk_text}".encode("utf-8"))
    return digest.hexdigest()


def build_document_chunks(file_path: Path) -> list[dict[str, Any]]:
    """Build chunk records with stable IDs and citation metadata."""
    document_chunks: list[dict[str, Any]] = []
    pages = extract_pdf_pages(file_path)
    for page in pages:
        chunks = chunk_text(page["text"])
        for index, chunk_text_value in enumerate(chunks):
            document_chunks.append(
                {
                    "source_name": page["source_name"],
                    "page_number": page["page_number"],
                    "chunk_index": index,
                    "chunk_id": stable_chunk_id(
                        page["source_name"],
                        page["page_number"],
                        index,
                        chunk_text_value,
                    ),
                    "text": chunk_text_value,
                }
            )
    return document_chunks


def list_pdf_files(directory: Path | str = PDF_DIR) -> list[Path]:
    """Return only PDFs from the configured directory."""
    directory_path = Path(directory)
    if not directory_path.exists():
        raise FileNotFoundError(f"PDF directory not found: {directory_path}")
    return sorted(
        [path for path in directory_path.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"],
        key=lambda item: item.name.lower(),
    )
