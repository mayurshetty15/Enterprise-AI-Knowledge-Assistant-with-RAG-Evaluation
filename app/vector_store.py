from __future__ import annotations

from functools import lru_cache
from typing import Any

import chromadb
import requests

from app.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    MAX_RETRIEVAL_RESULTS,
    OLLAMA_URL,
)


def _ollama_request(endpoint: str, payload: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
    response = requests.post(f"{OLLAMA_URL}{endpoint}", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def generate_embedding(text: str) -> list[float]:
    """Generate one embedding for a single text using Ollama's embedding model."""
    if not text or not text.strip():
        raise ValueError("Text cannot be empty for embedding")
    payload = {"model": EMBEDDING_MODEL, "input": text}
    try:
        data = _ollama_request("/api/embed", payload)
    except requests.HTTPError as error:
        if error.response is None or error.response.status_code != 404:
            raise
        data = _ollama_request(
            "/api/embeddings", {"model": EMBEDDING_MODEL, "prompt": text}
        )
    embedding = data.get("embedding")
    if isinstance(embedding, list) and all(isinstance(item, (int, float)) for item in embedding):
        return [float(item) for item in embedding]
    if isinstance(data.get("embeddings"), list) and data["embeddings"]:
        first = data["embeddings"][0]
        if isinstance(first, list) and first and all(isinstance(item, (int, float)) for item in first):
            return [float(item) for item in first]
    raise ValueError("Unexpected Ollama embedding response format")


@lru_cache(maxsize=1)
def get_collection() -> Any:
    """Create or get the persistent Chroma collection used for indexing."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def add_document_chunks(chunks: list[dict[str, Any]]) -> int:
    """Add new chunks to ChromaDB after skipping already-indexed chunk IDs."""
    if not chunks:
        return 0

    collection = get_collection()
    ids = [chunk["chunk_id"] for chunk in chunks]
    existing_ids = set(collection.get(ids=ids, include=[]).get("ids", []))
    pending = [chunk for chunk in chunks if chunk["chunk_id"] not in existing_ids]
    if not pending:
        return 0

    collection.add(
        ids=[chunk["chunk_id"] for chunk in pending],
        embeddings=[generate_embedding(chunk["text"]) for chunk in pending],
        documents=[chunk["text"] for chunk in pending],
        metadatas=[
            {
                "source_name": chunk["source_name"],
                "page_number": int(chunk["page_number"]),
                "chunk_index": int(chunk["chunk_index"]),
                **(
                    {"document_id": chunk["document_id"]}
                    if chunk.get("document_id")
                    else {}
                ),
            }
            for chunk in pending
        ],
    )
    return len(pending)


def delete_document_chunks(
    source_name: str | None = None, document_id: str | None = None
) -> int:
    """Delete all chunks belonging to a source document."""
    if (source_name is None) == (document_id is None):
        raise ValueError("Provide exactly one of source_name or document_id")
    collection = get_collection()
    field, value = (
        ("document_id", document_id) if document_id is not None else ("source_name", source_name)
    )
    results = collection.get(where={field: value}, include=[])
    ids = results.get("ids", [])
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def query_chunks(
    question: str,
    n_results: int = MAX_RETRIEVAL_RESULTS,
    document_ids: list[str] | None = None,
    source_names: list[str] | None = None,
    *,
    collection: Any | None = None,
) -> list[dict[str, Any]]:
    """Query ChromaDB for the most relevant chunks."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    collection = collection or get_collection()
    embedding = generate_embedding(question)
    filters: list[dict[str, Any]] = []
    if document_ids:
        filters.append({"document_id": {"$in": document_ids}})
    if source_names:
        filters.append({"source_name": {"$in": source_names}})
    where = None
    if len(filters) == 1:
        where = filters[0]
    elif len(filters) > 1:
        where = {"$or": filters}

    response = collection.query(
        query_embeddings=[embedding],
        n_results=min(max(1, n_results), 10),
        include=["documents", "metadatas", "distances"],
        where=where,
    )

    results: list[dict[str, Any]] = []
    for index in range(len(response.get("ids", [[]])[0])):
        meta = response.get("metadatas", [[{}]])[0][index]
        doc = response.get("documents", [[""]])[0][index]
        distance = response.get("distances", [[0.0]])[0][index]
        results.append(
            {
                "chunk_id": response.get("ids", [[]])[0][index],
                "text": doc,
                "source_name": meta.get("source_name", "unknown"),
                "page_number": int(meta.get("page_number", 1)),
                "distance": float(distance),
                "document_id": meta.get("document_id"),
            }
        )
    return results


def get_document_sources() -> list[str]:
    """List source filenames indexed in ChromaDB."""
    collection = get_collection()
    results = collection.get(include=["metadatas"])
    sources = []
    for metadata in results.get("metadatas", []):
        if metadata:
            source_name = metadata.get("source_name")
            if source_name and source_name not in sources:
                sources.append(source_name)
    return sources
