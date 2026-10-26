from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
GENERATION_MODEL = os.getenv("OLLAMA_GENERATION_MODEL", "gemma2:2b")

MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
MAX_RETRIEVAL_RESULTS = int(os.getenv("MAX_RETRIEVAL_RESULTS", "5"))
INSUFFICIENT_EVIDENCE_DISTANCE = float(os.getenv("INSUFFICIENT_EVIDENCE_DISTANCE", "0.85"))

# Allow replacement of a previous Chroma collection for a clean local rebuild.
COLLECTION_NAME = "enterprise_knowledge"


def ensure_project_paths() -> None:
    """Create required local directories if they do not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
