from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _project_path(environment_variable: str, default: str) -> Path:
    path = Path(os.getenv(environment_variable, default))
    return path if path.is_absolute() else PROJECT_ROOT / path


DATA_DIR = _project_path("DATA_DIRECTORY", "data")
PDF_DIR = _project_path("PDF_DIRECTORY", str(DATA_DIR / "pdfs"))
CHROMA_DIR = _project_path("CHROMA_DB_PATH", "chroma_db")
REGISTRY_PATH = _project_path(
    "DOCUMENT_REGISTRY_PATH", str(DATA_DIR / "documents.sqlite3")
)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
GENERATION_MODEL = os.getenv("OLLAMA_GENERATION_MODEL", "gemma2:2b")

MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
MAX_RETRIEVAL_RESULTS = int(os.getenv("MAX_RETRIEVAL_RESULTS", "5"))
INSUFFICIENT_EVIDENCE_DISTANCE = float(os.getenv("INSUFFICIENT_EVIDENCE_DISTANCE", "0.85"))
MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(10 * 1024 * 1024)))
FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]


def ollama_model_available(model_name: str, available_models: set[str]) -> bool:
    """Treat an untagged Ollama name as its default :latest tag."""
    return model_name in available_models or f"{model_name}:latest" in available_models


# Allow replacement of a previous Chroma collection for a clean local rebuild.
COLLECTION_NAME = "enterprise_knowledge"


def ensure_project_paths() -> None:
    """Create required local directories if they do not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
