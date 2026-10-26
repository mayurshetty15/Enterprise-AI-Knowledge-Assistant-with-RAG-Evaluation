from __future__ import annotations

import logging

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.config import MAX_UPLOAD_SIZE_BYTES
from app.schemas import (
    AskRequest,
    AskResponse,
    DeleteResponse,
    DocumentListResponse,
    EvaluationRunResponse,
    EvaluationRunsResponse,
    HealthResponse,
    UploadResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> dict:
    return await run_in_threadpool(request.app.state.health_check)


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(
    request: Request, file: UploadFile = File(...)
) -> JSONResponse:
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        await file.close()
        raise HTTPException(status_code=415, detail="Only PDF files are supported.")
    try:
        contents = await file.read(MAX_UPLOAD_SIZE_BYTES + 1)
    finally:
        await file.close()
    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="The PDF exceeds the maximum upload size.")
    try:
        document, already_indexed = await run_in_threadpool(
            request.app.state.document_service.upload, filename, contents
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logger.exception("PDF upload or indexing failed")
        raise HTTPException(
            status_code=503, detail="The PDF could not be indexed. Check local service availability."
        ) from error
    document["already_indexed"] = already_indexed
    return JSONResponse(
        status_code=200 if already_indexed else 201,
        content=UploadResponse(**document).model_dump(),
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(request: Request) -> dict:
    try:
        documents = await run_in_threadpool(request.app.state.document_service.list_documents)
    except Exception as error:
        logger.exception("Document listing failed")
        raise HTTPException(status_code=503, detail="The document index is unavailable.") from error
    return {"documents": documents}


@router.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document(request: Request, document_id: str) -> dict:
    try:
        deleted, deleted_chunks = await run_in_threadpool(
            request.app.state.document_service.delete, document_id
        )
    except Exception as error:
        logger.exception("Document deletion failed for %s", document_id)
        raise HTTPException(
            status_code=503, detail="The document could not be fully deleted."
        ) from error
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {
        "document_id": document_id,
        "deleted": True,
        "deleted_chunks": deleted_chunks,
    }


@router.post("/ask", response_model=AskResponse)
async def ask(request: Request, payload: AskRequest) -> dict:
    if not payload.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty.")
    try:
        return await run_in_threadpool(
            request.app.state.rag_service.ask,
            payload.question,
            payload.document_ids,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logger.exception("Question answering failed")
        raise HTTPException(
            status_code=503, detail="Question answering is unavailable. Check Ollama and the vector index."
        ) from error


@router.get("/evaluation/runs", response_model=EvaluationRunsResponse)
async def list_evaluation_runs(request: Request) -> dict:
    try:
        runs = await run_in_threadpool(request.app.state.evaluation_report_service.list_runs)
    except Exception as error:
        logger.exception("Evaluation report listing failed")
        raise HTTPException(status_code=503, detail="Evaluation reports are unavailable.") from error
    return {"runs": runs}


@router.get("/evaluation/runs/{run_id}", response_model=EvaluationRunResponse)
async def get_evaluation_run(request: Request, run_id: str) -> dict:
    try:
        run = await run_in_threadpool(
            request.app.state.evaluation_report_service.get_run, run_id
        )
    except Exception as error:
        logger.exception("Evaluation report read failed")
        raise HTTPException(status_code=503, detail="The evaluation report is unavailable.") from error
    if run is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found.")
    return {"run": run}
