# Enterprise AI Knowledge Assistant with RAG Evaluation

This project is a beginner-friendly, local-first RAG pipeline for answering questions from company documents. Phase 1 focuses on the local environment, PDF ingestion, vector storage, retrieval, answer generation, and simple validation.

## Goals for Phase 1

- Verify Python, Node, Git, and Ollama
- Create a Python 3.12 virtual environment
- Install dependencies into the project-local environment
- Load PDFs from `data/pdfs/`
- Split PDF text into overlapping chunks
- Embed text with Ollama's `nomic-embed-text` model
- Store embeddings in ChromaDB locally
- Retrieve relevant chunks for a question
- Generate grounded answers with `gemma2:2b`
- Return an insufficient-evidence response when the retrieved facts are weak

## Project structure

```text
enterprise-ai-knowledge-assistant/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── ingestion.py
│   ├── vector_store.py
│   ├── retrieval.py
│   ├── generation.py
│   ├── rag_pipeline.py
│   ├── cli.py
│   └── __init__.py
├── data/
│   └── pdfs/
├── tests/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
└── .venv/
```

## Windows setup

From PowerShell in the project root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Verify Ollama models

```powershell
ollama list
ollama run nomic-embed-text "hello"
ollama run gemma2:2b "Say hello in one short sentence."
```

## Index PDFs

Put your PDFs in `data/pdfs/`, then run:

```powershell
python -m app.cli ingest
```

## Ask a question

```powershell
python -m app.cli ask --question "What is the company's work-from-home policy?"
```

## Run tests

```powershell
python -m pytest -q
```

## Important notes

- Do not push real secrets or private documents to GitHub.
- The project is local-first and does not yet include FastAPI endpoints or a Vercel deployment.
- ChromaDB is stored locally in the `chroma_db/` folder.
- Ollama must be running locally for embeddings and generation.
