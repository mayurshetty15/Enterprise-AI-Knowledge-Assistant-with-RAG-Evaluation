from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import FRONTEND_ORIGINS
from app.services.document_service import DocumentService, check_services
from app.services.evaluation_reports import EvaluationReportService
from app.services.rag_service import RAGService


def create_app(
    document_service: DocumentService | None = None,
    rag_service: RAGService | None = None,
    health_check: Callable[[], dict[str, Any]] = check_services,
) -> FastAPI:
    documents = document_service or DocumentService()
    application = FastAPI(
        title="Enterprise AI Knowledge Assistant",
        version="2.0.0",
        description="Local-first document question answering with the existing RAG pipeline.",
    )
    application.state.document_service = documents
    application.state.rag_service = rag_service or RAGService(documents)
    application.state.evaluation_report_service = EvaluationReportService()
    application.state.health_check = health_check
    application.add_middleware(
        CORSMiddleware,
        allow_origins=FRONTEND_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )
    application.include_router(router)
    return application


app = create_app()
