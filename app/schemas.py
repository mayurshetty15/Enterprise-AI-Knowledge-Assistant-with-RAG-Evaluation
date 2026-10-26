from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    services: dict[str, object]


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    status: str
    created_at: str | None = None


class UploadResponse(DocumentResponse):
    already_indexed: bool = False


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]


class DeleteResponse(BaseModel):
    document_id: str
    deleted: bool
    deleted_chunks: int


class AskRequest(BaseModel):
    question: str = Field(max_length=4000)
    document_ids: list[str] | None = None


class SourceResponse(BaseModel):
    document_id: str
    filename: str
    page: int
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    retrieval_count: int
    status: str


class EvaluationRunsResponse(BaseModel):
    runs: list[dict[str, object]]


class EvaluationRunResponse(BaseModel):
    run: dict[str, object]
