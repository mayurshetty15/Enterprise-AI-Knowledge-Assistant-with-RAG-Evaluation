# Enterprise AI Knowledge Assistant with RAG Evaluation

A local-first knowledge assistant for asking questions about PDF documents. Phase 1 provides the existing PDF ingestion, Ollama embeddings, ChromaDB retrieval, and locally generated answers. Phase 2 exposes that pipeline through FastAPI and adds a responsive React/Vite user interface. No hosted LLM API or paid service is required.

## Architecture

- `app/ingestion.py` extracts page-level text and creates overlapping chunks.
- `app/vector_store.py` creates embeddings with Ollama and stores chunks in the existing persistent `enterprise_knowledge` ChromaDB collection.
- `app/retrieval.py`, `app/generation.py`, and `app/rag_pipeline.py` provide the Phase 1 retrieval and answer-generation flow. The API reuses these modules.
- `app/services/document_service.py` tracks uploads and indexing status in a local SQLite registry at `data/documents.sqlite3`.
- `app/main.py` and `app/api/routes.py` expose the HTTP API.
- `frontend/` contains the React/Vite application.
- `data/pdfs/` remains available for the Phase 1 CLI workflow. Uploaded PDF bytes are processed in a temporary file and removed after indexing; the registry stores document metadata and ChromaDB stores the chunks.

Phase 1 documents already present in ChromaDB remain available to the API. They are identified by their source filename; newly uploaded documents also have unique document IDs.
When legacy Phase 1 chunks are first listed, the API adds a stable document ID to their existing metadata so selection and deletion remain scoped even when filenames overlap.

## Technology

- Python 3.12, FastAPI, Pydantic, Uvicorn
- React 18 and Vite
- Ollama with `nomic-embed-text` and `gemma2:2b`
- ChromaDB persistent local vector storage
- PyMuPDF PDF text extraction
- SQLite document metadata registry

## Prerequisites and local model setup

Install Python 3.12, Node.js 18 or later/npm, and [Ollama](https://ollama.com/). Start Ollama, then pull the configured models:

```powershell
ollama pull nomic-embed-text
ollama pull gemma2:2b
ollama list
```

The app uses `http://localhost:11434` by default. It does not automatically load `.env`; set environment variables in the PowerShell session before starting a process if you need to override defaults. See [`.env.example`](./.env.example) for supported backend variables and [`frontend/.env.example`](./frontend/.env.example) for the frontend API URL.

## Backend setup and startup (Windows PowerShell)

Run these commands from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The API and OpenAPI documentation are available at `http://127.0.0.1:8000` and `http://127.0.0.1:8000/docs`.

Optional backend configuration:

```powershell
$env:OLLAMA_URL = "http://localhost:11434"
$env:OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
$env:OLLAMA_GENERATION_MODEL = "gemma2:2b"
$env:CHROMA_DB_PATH = "chroma_db"
$env:DOCUMENT_REGISTRY_PATH = "data/documents.sqlite3"
$env:MAX_UPLOAD_SIZE_BYTES = "10485760"
$env:FRONTEND_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"
```

Relative data and database paths are resolved from the repository root. The defaults preserve the Phase 1 ChromaDB location and collection.

## Frontend setup and startup

In a second PowerShell window:

```powershell
Set-Location frontend
npm install
npm run dev
```

Open the Vite URL printed in the terminal (normally `http://localhost:5173`). To use a different API address, copy `frontend/.env.example` to `frontend/.env.local`, set `VITE_API_BASE_URL`, and restart Vite. Frontend environment variables are public configuration only; do not put secrets in them.

## Using the application

1. Start Ollama, the FastAPI backend, and the Vite frontend.
2. Upload a text-based PDF by dropping it on the upload panel or selecting it. The API validates the PDF and the configured maximum size (10 MB by default), extracts text, and indexes the chunks in the existing collection.
3. Select one or more documents to scope a question, or leave them unchecked to search the full index.
4. Ask a question. The assistant displays the answer separately from page-level source evidence. Expand a source card to read its supporting snippet.
5. Delete a document from the knowledge-base list after confirming. This removes only that uploaded document's chunks and registry record.

Repeated uploads of identical file content are detected and do not create a second index entry.

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Reports API, Ollama, configured models, and vector database status. A degraded status means at least one dependency is unavailable. |
| `POST` | `/api/documents/upload` | Upload and index one PDF as multipart form field `file`. |
| `GET` | `/api/documents` | List registered uploads and documents already in the Phase 1 collection. |
| `DELETE` | `/api/documents/{document_id}` | Delete one selected document and its chunks. |
| `POST` | `/api/ask` | Answer a question, optionally scoped to document IDs. |
| `GET` | `/api/evaluation/runs` | List safe summaries of previously generated local evaluation reports. |
| `GET` | `/api/evaluation/runs/{run_id}` | Read one report through a fixed report directory; no arbitrary path access. |

Example health request:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

Example upload:

```powershell
curl.exe -F "file=@C:\path\to\report.pdf" http://127.0.0.1:8000/api/documents/upload
```

Example question across the full collection:

```powershell
$body = @{ question = "What are the main findings in the uploaded document?"; document_ids = @() } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/ask -Method Post -ContentType "application/json" -Body $body
```

To limit retrieval, pass IDs returned by `GET /api/documents`, for example:

```json
{
  "question": "What are the main findings?",
  "document_ids": ["document-id-from-the-list"]
}
```

An answer response includes `answer`, `sources` (document ID, filename, page, and retrieved evidence snippet), `retrieval_count`, and `status`. With insufficient evidence it returns an explicit insufficiency message and no fabricated sources. Document text is treated as untrusted evidence, not as instructions for the model.

## Phase 1 CLI

The original command-line workflows remain available:

```powershell
python -m app.cli check
python -m app.cli ingest
python -m app.cli ask --question "What is the company's work-from-home policy?"
```

Place Phase 1 PDFs under `data/pdfs/` before running `ingest`.

## Tests and frontend build

Run backend tests (they use temporary registries and mocked vector/Ollama interactions):

```powershell
python -m pytest -q
```

Build the frontend:

```powershell
Set-Location frontend
npm run build
```

## RAG evaluation

The evaluation package reuses the Phase 1 Ollama embedding, retrieval, and
generation functions. Each run creates a uniquely named collection in an
in-memory `chromadb.EphemeralClient`, indexes only the corpus from the selected
evaluation dataset, and never reads, mutates, or deletes the persistent
`enterprise_knowledge` collection. The runner is a synchronous CLI so evaluation
does not block the FastAPI event loop. FastAPI exposes only read-only report
listing and detail endpoints, and the React Evaluation page reads those reports.

### Dataset schema and ground truth

The version-controlled starter dataset is
[`evaluation/datasets/synthetic-v1.json`](./evaluation/datasets/synthetic-v1.json).
It contains three synthetic cases (factual lookup, multi-page question, and
insufficient evidence) over synthetic page text. These facts and results are
not real organizational knowledge. See
[`evaluation/datasets/README.md`](./evaluation/datasets/README.md) for
human-verification and privacy requirements before adding cases from PDFs.

Each JSON dataset has a `dataset_version`, `description`, `corpus`, and `cases`.
Corpus documents contain a stable `document_id`, basename-only `filename`, and
pages with positive page numbers and text. Each case requires a unique `case_id`,
question, category, `answerable`, reference answer (null for unanswerable
cases), and expected source document/page pairs. Optional
`relevant_chunk_ids` enables exact chunk-level retrieval scoring only when those
IDs have been independently verified. Real-document answer text, document
identity, and page numbers must be checked by a human against the original PDF.
Do not commit private PDFs, page text, or private ground truth.

### Metrics and interpretation

Retrieval relevance is a unique source-page pair
`(document_id, page)` unless exact `relevant_chunk_ids` are provided. Duplicate
retrieved chunk IDs are discarded; in page mode, multiple chunks from the same
page count once. For each K (1, 3, 5):

- **Precision@K** = relevant retrieved units in the first K / `min(K, number
  of unique retrieved units)`. When relevance is known but nothing was
  retrieved, the score is 0.
- **Recall@K** = relevant retrieved units in the first K / all labeled
  relevant units.
- **MRR** = reciprocal rank of the first relevant retrieved unit, or 0 when
  there is no relevant retrieved unit.
- **Hit Rate@K** = 1 if any relevant unit occurs in the first K, otherwise 0.
- **Document accuracy@K** = expected documents represented in the top K /
  unique expected documents.
- **Page accuracy@K** = expected source pages recovered in the top K /
  unique expected source pages.

Cases with no labeled relevant sources have retrieval scores of `null` (not
zero), because recall and accuracy have no valid denominator. In chunk-label
mode, precision, recall, hit rate, and MRR use chunk units; document and page
accuracy still require expected source labels.

Answer quality uses transparent deterministic proxies: **answer correctness**
is multiset token F1 against the reference; **lexical faithfulness** is the
fraction of meaningful answer tokens found in retrieved context; and **lexical
answer relevance** is the fraction of meaningful question terms echoed in the
answer. These lexical measures are not semantic ground truth. **Abstention
correctness** checks whether the pipeline status matches the case's answerable
label. **Citation F1** compares exact `(document_id, page)` source sets; an
empty expected and actual citation set scores 1, as appropriate for an
unanswerable case. A local Ollama judge is optional; it scores correctness,
faithfulness, and relevance from 1 to 5 with temperature 0 and seed 0. Those
scores are subjective, are labeled as such, and are not combined with the
deterministic scores.

### Run and inspect evaluations

From the repository root, with Ollama running and the configured models
available:

```powershell
python -m evaluation.cli
```

To include the optional local-model judge:

```powershell
python -m evaluation.cli --llm-judge
```

To run a different prepared dataset or write reports elsewhere:

```powershell
python -m evaluation.cli --dataset evaluation/datasets/my-dataset.json --reports-dir evaluation/reports
```

Each run creates a unique directory under `evaluation/reports/` containing:

- `report.json`: per-case results, aggregate metrics, model/configuration and
  dataset hash metadata, latency summaries, and failures.
- `cases.csv`: one row per case for analysis.
- `report.md`: readable aggregate report and failure examples.

The report directory is ignored by Git; previous run directories are retained.
Reports intentionally omit questions, corpus text, reference answers, and
generated answer text. The dashboard lists and compares available report runs;
the CLI is the only evaluation trigger.

Retrieval, generation, and total latencies use a monotonic timer. Median is
reported for available observations; p95 is reported only with at least 20
samples and the sample count is always included. A small dataset is a
functional smoke evaluation, not a statistically reliable benchmark. Model
versions, chunking/retrieval settings, K values, dataset SHA-256, timestamp, and
local-judge settings are included in report metadata. Total query latency covers retrieval and answer generation, excluding
corpus indexing and evaluation scoring. Local Ollama inference has no applicable
API/cloud cost.

To compare configurations, keep the dataset version fixed, change only the
desired environment/configuration value (for example
`OLLAMA_GENERATION_MODEL`), then run the same CLI again. Reports receive unique
IDs and remain side by side for dashboard comparison. Check each run's
configuration metadata because changing multiple settings at once prevents
isolating the cause of metric differences.

Known limitations: the bundled corpus is synthetic and deliberately small;
lexical answer-quality scores are rough proxies; local-model judge scores can
vary despite deterministic generation settings; evaluation quality depends on
human-verified labels; and reported query latency excludes corpus embedding and
initial indexing time. The evaluator uses the existing RAG behavior and does
not claim the answer is objectively correct merely because a score is high.

## Troubleshooting

- **Ollama is unavailable:** Start Ollama and check `ollama list`; verify `OLLAMA_URL`.
- **A model reports missing:** Pull the exact configured tag, e.g. `ollama pull gemma2:2b`, and verify the corresponding environment variable.
- **Evaluation embedding/model errors:** Verify Ollama is reachable and both the configured embedding and generation models are listed with `ollama list`. Evaluation indexes its synthetic/prepared corpus in memory and does not require a production Chroma reset.
- **The API reports a degraded vector database:** Check that `CHROMA_DB_PATH` points to the intended local index and that the process can read/write it. Do not delete `chroma_db/` to fix this.
- **The frontend cannot reach the backend:** Confirm the API is running at `127.0.0.1:8000`, check `VITE_API_BASE_URL`, and include the frontend origin in `FRONTEND_ORIGINS`.
- **PDF upload fails:** Only PDFs with readable embedded text are supported; password-protected, corrupt, empty, scanned-image-only, and oversized files are rejected. OCR is not included.
- **Answers have no evidence:** Check that the documents are indexed, that selected document IDs are correct, and that the question is supported by retrieved passages.

## Security and limitations

- The application is designed for local use. It has no authentication or authorization; do not expose the API or Ollama port to untrusted networks.
- CORS defaults to the two local Vite development origins. Configure only trusted origins if exposing the service.
- Upload size is limited by `MAX_UPLOAD_SIZE_BYTES`. PDF bytes are staged in a temporary file and cleaned up after processing; uploaded files are not retained by the application.
- ChromaDB, SQLite metadata, and model data remain local. `.gitignore` excludes the local database, PDFs, `.env` files, and frontend build/dependency artifacts.
- PDF text extraction does not include OCR for scanned pages. Model output can still be imperfect; verify answers against the displayed evidence.
