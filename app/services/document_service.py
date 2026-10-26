from __future__ import annotations

import hashlib
import logging
import sqlite3
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import fitz

from app.config import REGISTRY_PATH
from app.ingestion import build_document_chunks
from app.vector_store import add_document_chunks, delete_document_chunks, get_collection

logger = logging.getLogger(__name__)


class DocumentService:
    """Manage uploaded documents and their records alongside the shared Chroma index."""

    def __init__(self, registry_path: Path | str = REGISTRY_PATH):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL UNIQUE,
                    page_count INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.registry_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _document(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "document_id": row["document_id"],
            "filename": row["filename"],
            "source_name": row["filename"],
            "page_count": int(row["page_count"]),
            "chunk_count": int(row["chunk_count"]),
            "status": row["status"],
            "created_at": row["created_at"],
        }

    def upload(self, filename: str, contents: bytes) -> tuple[dict[str, Any], bool]:
        if not contents:
            raise ValueError("The uploaded PDF is empty.")

        digest = hashlib.sha256(contents).hexdigest()
        with self._connection() as connection:
            previous = connection.execute(
                "SELECT * FROM documents WHERE content_sha256 = ?", (digest,)
            ).fetchone()
        if previous:
            if previous["status"] != "failed":
                return self._document(previous), True
            delete_document_chunks(document_id=previous["document_id"])
            with self._connection() as connection:
                connection.execute(
                    "DELETE FROM documents WHERE document_id = ?",
                    (previous["document_id"],),
                )

        temp_path: Path | None = None
        document_id = str(uuid.uuid4())
        registered = False
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temporary:
                temporary.write(contents)
                temp_path = Path(temporary.name)

            try:
                with fitz.open(temp_path) as pdf:
                    if pdf.is_encrypted:
                        raise ValueError("Password-protected PDFs are not supported.")
                    page_count = pdf.page_count
                    if page_count == 0:
                        raise ValueError("The PDF does not contain any pages.")
            except ValueError:
                raise
            except (RuntimeError, fitz.FileDataError) as error:
                raise ValueError("The uploaded file is not a readable PDF.") from error

            chunks = build_document_chunks(temp_path)
            if not chunks:
                raise ValueError("No readable text was found in the PDF.")

            for index, chunk in enumerate(chunks):
                chunk["source_name"] = Path(filename).name
                chunk["document_id"] = document_id
                chunk["chunk_id"] = hashlib.sha256(
                    f"{document_id}:{index}:{chunk['chunk_id']}".encode("utf-8")
                ).hexdigest()

            try:
                with self._connection() as connection:
                    connection.execute(
                        """
                        INSERT INTO documents
                            (document_id, filename, content_sha256, page_count, status)
                        VALUES (?, ?, ?, ?, 'indexing')
                        """,
                        (document_id, Path(filename).name, digest, page_count),
                    )
            except sqlite3.IntegrityError:
                with self._connection() as connection:
                    previous = connection.execute(
                        "SELECT * FROM documents WHERE content_sha256 = ?", (digest,)
                    ).fetchone()
                if previous:
                    return self._document(previous), True
                raise
            registered = True

            indexed_count = add_document_chunks(chunks)
            if indexed_count != len(chunks):
                raise RuntimeError("Not all PDF chunks were added to the vector index.")

            with self._connection() as connection:
                connection.execute(
                    "UPDATE documents SET chunk_count = ?, status = 'indexed' WHERE document_id = ?",
                    (indexed_count, document_id),
                )
                row = connection.execute(
                    "SELECT * FROM documents WHERE document_id = ?", (document_id,)
                ).fetchone()
            return self._document(row), False
        except Exception:
            if registered:
                try:
                    delete_document_chunks(document_id=document_id)
                except Exception:
                    logger.exception("Failed to remove partially indexed document %s", document_id)
                with self._connection() as connection:
                    connection.execute(
                        "UPDATE documents SET status = 'failed' WHERE document_id = ?",
                        (document_id,),
                    )
            raise
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def list_documents(self) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY created_at DESC, filename COLLATE NOCASE"
            ).fetchall()
        documents = [self._document(row) for row in rows]
        collection = get_collection()
        collection_data = collection.get(include=["metadatas"])
        chunk_ids = collection_data.get("ids", [])
        metadatas = collection_data.get("metadatas", [])
        legacy_updates: list[tuple[str, dict[str, Any]]] = []
        for chunk_id, metadata in zip(chunk_ids, metadatas):
            if metadata and not metadata.get("document_id") and metadata.get("source_name"):
                legacy_id = "legacy:" + hashlib.sha256(
                    metadata["source_name"].encode("utf-8")
                ).hexdigest()[:24]
                legacy_updates.append(
                    (chunk_id, {**metadata, "document_id": legacy_id})
                )
        batch_size = 5000
        for offset in range(0, len(legacy_updates), batch_size):
            batch = legacy_updates[offset : offset + batch_size]
            collection.update(
                ids=[chunk_id for chunk_id, _ in batch],
                metadatas=[metadata for _, metadata in batch],
            )
        if legacy_updates:
            legacy_metadata = {
                chunk_id: metadata for chunk_id, metadata in legacy_updates
            }
            metadatas = [
                legacy_metadata.get(chunk_id, metadata)
                for chunk_id, metadata in zip(chunk_ids, metadatas)
            ]

        legacy: dict[str, dict[str, Any]] = {}
        for metadata in metadatas:
            if not metadata or not str(metadata.get("document_id", "")).startswith("legacy:"):
                continue
            filename = metadata.get("source_name")
            if not filename:
                continue
            entry = legacy.setdefault(
                filename,
                {
                    "document_id": metadata["document_id"],
                    "filename": filename,
                    "source_name": filename,
                    "page_count": 0,
                    "chunk_count": 0,
                    "status": "indexed",
                    "created_at": None,
                },
            )
            entry["chunk_count"] += 1
            entry["page_count"] = max(
                entry["page_count"], int(metadata.get("page_number", 0))
            )
        return documents + sorted(legacy.values(), key=lambda item: item["filename"].lower())

    def resolve_document_ids(self, document_ids: list[str]) -> tuple[list[str], list[str]]:
        if not document_ids:
            return [], []
        legacy_ids = [item for item in document_ids if item.startswith("legacy:")]
        registered_ids = [item for item in document_ids if not item.startswith("legacy:")]
        found: set[str] = set()

        for offset in range(0, len(registered_ids), 500):
            batch = registered_ids[offset : offset + 500]
            placeholders = ",".join("?" for _ in batch)
            with self._connection() as connection:
                rows = connection.execute(
                    f"SELECT document_id FROM documents WHERE status = 'indexed' "
                    f"AND document_id IN ({placeholders})",
                    batch,
                ).fetchall()
            found.update(row["document_id"] for row in rows)

        for offset in range(0, len(legacy_ids), 500):
            batch = legacy_ids[offset : offset + 500]
            matches = get_collection().get(
                where={"document_id": {"$in": batch}},
                include=["metadatas"],
            )
            found.update(
                metadata["document_id"]
                for metadata in matches.get("metadatas", [])
                if metadata and metadata.get("document_id") in batch
            )

        missing = [document_id for document_id in document_ids if document_id not in found]
        if missing:
            raise ValueError("One or more selected documents do not exist or are not indexed.")
        return document_ids, []

    def delete(self, document_id: str) -> tuple[bool, int]:
        if document_id.startswith("legacy:"):
            matches = get_collection().get(
                where={"document_id": document_id},
                include=[],
            )
            if not matches.get("ids"):
                return False, 0
            deleted_chunks = delete_document_chunks(document_id=document_id)
            return True, deleted_chunks

        with self._connection() as connection:
            row = connection.execute(
                "SELECT document_id FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
        if row is None:
            return False, 0

        deleted_chunks = delete_document_chunks(document_id=document_id)
        with self._connection() as connection:
            connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
        return True, deleted_chunks


def check_services() -> dict[str, Any]:
    """Return explicit health states for Ollama models and the persistent vector index."""
    import requests

    from app.config import (
        EMBEDDING_MODEL,
        GENERATION_MODEL,
        OLLAMA_URL,
        ollama_model_available,
    )

    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        response.raise_for_status()
        models = {
            item.get("name")
            for item in response.json().get("models", [])
            if isinstance(item, dict)
        }
        ollama = {"status": "ok", "reachable": True}
        chat_model = {
            "status": "available"
            if ollama_model_available(GENERATION_MODEL, models)
            else "missing",
            "name": GENERATION_MODEL,
        }
        embedding_model = {
            "status": "available"
            if ollama_model_available(EMBEDDING_MODEL, models)
            else "missing",
            "name": EMBEDDING_MODEL,
        }
    except (requests.RequestException, ValueError, TypeError):
        ollama = {"status": "unavailable", "reachable": False}
        chat_model = {"status": "unavailable", "name": GENERATION_MODEL}
        embedding_model = {"status": "unavailable", "name": EMBEDDING_MODEL}

    try:
        get_collection().count()
        vector_database = {"status": "ok"}
    except Exception:
        logger.exception("Vector database health check failed")
        vector_database = {"status": "unavailable"}

    services = {
        "api": "ok",
        "ollama": ollama,
        "chat_model": chat_model,
        "embedding_model": embedding_model,
        "vector_database": vector_database,
    }
    statuses = [
        ollama["status"],
        chat_model["status"],
        embedding_model["status"],
        vector_database["status"],
    ]
    return {
        "status": "ok" if all(status in ("ok", "available") for status in statuses) else "degraded",
        "services": services,
    }
