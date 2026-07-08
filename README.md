# alef_education

AI learning assistant for textbook PDFs using FastAPI, LangGraph, Bedrock models, semantic chunking, and local FAISS retrieval.

## Repository Overview

This repository implements a local-first Retrieval-Augmented Generation (RAG) workflow:

1. Ingest a PDF via API.
2. Extract and clean page text.
3. Build semantic chunks with page attribution.
4. Create embeddings and persist FAISS index + metadata.
5. Answer student questions with citation-aware responses.

Main directories:

- `app/`: backend code (API, ingestion, retrieval, QA orchestration).
- `outputs/`: extracted JSON and session artifacts.
- `vectorstores/`: persisted FAISS indexes and chunk metadata.
- `docs/`: source assignment and sample PDF.
- `scripts/`: local evaluation utilities.

Key files:

- `app/main.py`: FastAPI entrypoint.
- `app/routers/ingest.py`: `/ingest` endpoint.
- `app/routers/qa.py`: `/ask` endpoint.
- `app/routers/health.py`: `/health` endpoint.
- `streamlit_app.py`: Streamlit UI for health, ingest, and ask flows.

## API Endpoints

### `GET /health`
Purpose: service status check.

Example:

```bash
curl http://127.0.0.1:8000/health
```

### `POST /ingest`
Purpose: upload a PDF, extract/clean text, build embeddings, and persist vector artifacts.

Request:
- `multipart/form-data`
- field name: `file`

Example:

```bash
curl -X POST http://127.0.0.1:8000/ingest \
	-F "file=@docs/A Quick Algebra Review.pdf"
```

Response (shape):

```json
{
	"message": "PDF ingested and embeddings built successfully.",
	"output_json": "outputs/...json",
	"total_pages": 31,
	"index_path": "vectorstores/...faiss",
	"metadata_path": "vectorstores/...metadata.json",
	"chunk_count": 33,
	"model_id": "amazon.titan-embed-text-v2:0"
}
```

### `POST /ask`
Purpose: answer a student question using local retrieval + LLM synthesis.

Request example:

```json
{
	"student_id": "student-123",
	"question": "How do I solve 4x - 7 = 21?",
	"grade_level": "grade_8"
}
```

Response example:

```json
{
	"request_id": "req-001",
	"answer": "Add 7 to both sides to get 4x = 28. Then divide both sides by 4, so x = 7.",
	"sources": [
		{
			"source_id": "A_Quick_Algebra_Review_20260708T172823Z.json#page=4",
			"title": "Solving Equations",
			"page": 4,
			"page_start": 4,
			"page_end": 4,
			"excerpt": "..."
		}
	],
	"safety_status": "answered"
}
```

## How To Run

Prerequisites:

- Python virtual environment exists at `.venv`.
- Dependencies are installed from `requirement.txt`.
- AWS/Bedrock credentials are configured in your environment (for embedding + LLM calls).

Install dependencies (if needed):

```bash
.venv/bin/pip install -r requirement.txt
```

### Run backend (FastAPI)

Use the exact command already used in this repo:

```bash
pids=$(lsof -ti tcp:8000); [ -n "$pids" ] && kill -9 $pids; /Users/swapnanilsharmah/Documents/alef_education/.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Backend will be available at `http://127.0.0.1:8000`.

### Run Streamlit UI

Use the exact command already used in this repo:

```bash
/Users/swapnanilsharmah/Documents/alef_education/.venv/bin/streamlit run streamlit_app.py
```

Then open the URL printed by Streamlit (typically `http://localhost:8501`).

## Typical Local Workflow

1. Start backend.
2. Start Streamlit UI.
3. Open UI and verify `Health` tab.
4. Ingest a PDF in `Ingest PDF` tab.
5. Ask questions in `Ask` tab and inspect returned sources.

## Evaluation Script

Run chunking quality checks without external API calls:

```bash
.venv/bin/python scripts/evaluate_chunking.py outputs/A_Quick_Algebra_Review_20260708T172823Z.json --show-examples 10
```

This validates:

- chunk count and average size,
- cross-page chunk presence,
- source span presence,
- page range validity,
- empty chunk checks.