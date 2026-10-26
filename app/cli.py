from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from app.config import (
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    PDF_DIR,
    ollama_model_available,
)
from app.rag_pipeline import RAGPipeline


def _run_command(command: list[str]) -> tuple[int, str, str]:
    process = subprocess.run(command, capture_output=True, text=True)
    return process.returncode, process.stdout.strip(), process.stderr.strip()


def check_environment() -> dict:
    """Check for required Python packages, the virtual environment, and Ollama models."""
    python_version = sys.version.replace("\n", " ")
    venv_active = sys.prefix != sys.base_prefix
    package_names = ["chromadb", "fitz", "requests", "fastapi", "uvicorn", "multipart"]
    installed = {}
    for module_name in package_names:
        try:
            __import__(module_name)
            installed[module_name] = True
        except Exception:
            installed[module_name] = False

    return_code, ollama_stdout, ollama_stderr = _run_command(["ollama", "list"])
    models = []
    if return_code == 0:
        for line in ollama_stdout.splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 2:
                models.append(parts[0])

    available_models = set(models)
    model_status = {
        "embedding": ollama_model_available(EMBEDDING_MODEL, available_models),
        "generation": ollama_model_available(GENERATION_MODEL, available_models),
    }

    print("Environment check")
    print(f"Python: {python_version}")
    print(f"Virtual environment active: {venv_active}")
    print(f"Packages installed: {installed}")
    print(f"Ollama models available: {model_status}")
    print(f"Ollama output: {ollama_stdout[:200]}")
    if ollama_stderr:
        print(f"Ollama errors: {ollama_stderr}")
    return {"python": python_version, "venv_active": venv_active, "packages": installed, "models": model_status}


def ingest_documents() -> None:
    """Index all PDFs in the configured folder."""
    pipeline = RAGPipeline(pdf_dir=PDF_DIR)
    pdf_directory = Path(PDF_DIR)
    if not pdf_directory.exists():
        raise FileNotFoundError(f"PDF directory does not exist: {pdf_directory}")

    indexed = pipeline.ingest_documents()
    if not indexed:
        print(f"No PDFs were indexed. Place PDF files in {pdf_directory}.")
        return
    print(f"Indexed {len(indexed)} PDF file(s): {', '.join(indexed)}")


def ask_question(question: str | None = None) -> None:
    """Ask one question interactively or using the --question argument."""
    pipeline = RAGPipeline(pdf_dir=PDF_DIR)

    if question is None:
        question = input("Enter your question: ").strip()

    result = pipeline.ask(question)
    print("\nQuestion:", result["question"])
    print("\nAnswer:")
    print(result["answer"])
    if result.get("citations"):
        print("\nCitations:")
        for citation in result["citations"]:
            print(f"- {citation['source_name']} (page {citation['page_number']})")
    else:
        print("\nNo citations available because the evidence was insufficient.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enterprise AI Knowledge Assistant - Phase 1")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check", help="Check Python, packages, and Ollama model availability")
    subparsers.add_parser("ingest", help="Index PDFs found in data/pdfs")
    ask_parser = subparsers.add_parser("ask", help="Ask a question from the indexed knowledge base")
    ask_parser.add_argument("--question", help="Optional question to answer without interactive input")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "check":
        check_environment()
    elif args.command == "ingest":
        ingest_documents()
    elif args.command == "ask":
        ask_question(args.question)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
