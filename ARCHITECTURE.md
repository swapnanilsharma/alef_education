# Architecture

## System Overview
This project is a local-first Retrieval-Augmented Generation (RAG) assistant for textbook-style PDFs.

Core flow:
1. Upload PDF to `/ingest`.
2. Extract and clean page text.
3. Build semantic chunks with page-source spans.
4. Generate embeddings and persist FAISS + metadata artifacts.
5. Ask questions via `/ask`.
6. Run LangGraph workflow: safety -> retrieval -> confidence gate -> answer synthesis.

Main components:
- API layer: FastAPI routers in `app/routers`.
- Ingestion: PDF extraction, cleaning, semantic chunking, embedding pipeline in `app/ingestion`.
- Retrieval: Bedrock embeddings + FAISS index search in `app/retrieval`.
- Orchestration: LangGraph workflow in `app/qa/graph.py`.
- UI: Streamlit client in `streamlit_app.py`.

## Design Choices

### 1) Semantic chunking with page attribution
Choice:
- Chunk by semantic blocks (headings, examples, practice/equation groups) rather than only fixed character windows.
- Preserve source traceability with `page_start`, `page_end`, and per-page `source_spans`.

Why:
- Textbooks often include layout noise, formulas, and examples that do not align to page boundaries.
- Better retrieval precision requires smaller, topic-coherent chunks.

Trade-off:
- More logic and heuristics in chunking.
- Heading detection may still need tuning for unusual PDFs.

### 2) Local FAISS for retrieval
Choice:
- Persist embeddings in FAISS (`IndexFlatIP`) with normalized vectors and JSON metadata.

Why:
- Simple, fast local setup.
- No managed vector DB dependency for the assignment.

Trade-off:
- No built-in filtering, hybrid search, or reranking unless explicitly added.

### 3) LangGraph for QA orchestration
Choice:
- Use a graph workflow with explicit steps: safety classification, retrieval, confidence gating, answer generation, and logging.

Why:
- Clear control flow and easier extension.
- Easy insertion points for policy and observability.

Trade-off:
- Slightly more framework complexity than a single function pipeline.

### 4) Strict source-grounded answering
Choice:
- Prompt and confidence gate enforce: no hallucinated citations/facts; return insufficient when context is weak.

Why:
- Matches assignment safety and citation requirements.

Trade-off:
- Can be conservative when retrieval score is low.

## PDF Ingestion Approach
1. Parse uploaded PDF bytes page by page.
2. Clean extraction noise while preserving math content.
3. Build semantic blocks from cleaned text.
4. Pack blocks into bounded chunks using configurable targets and overlap.
5. Embed chunks.
6. Persist FAISS index and chunk metadata JSON.

Artifacts:
- Cleaned extraction JSON in `outputs/`.
- Vector index + metadata in `vectorstores/`.

## Failure Modes and Handling

### Ingest-time failures
- Invalid/non-PDF input -> `400`.
- Empty uploads -> `400`.
- PDF parse failures -> `400`.
- Embedding/runtime provider failures -> `502`.
- No usable text/chunks -> `400`.

### Ask-time failures
- Missing vector artifacts -> `404`.
- Invalid request payload -> `400`.
- Retrieval/generation runtime failures -> `502`.

### Quality-related failure modes
- Noisy OCR-like pages may degrade chunking.
- Weak heading detection can reduce source title quality.
- Pure semantic vector search may miss lexical edge-cases without hybrid retrieval.
- Very long formulas/tables may still produce suboptimal chunks.

## Scaling Approach

### Near-term (single node)
- Keep local FAISS and file-based artifacts.
- Add background ingest workers if upload volume grows.
- Cache/reuse embedding clients to reduce initialization overhead.

### Mid-term
- Partition indexes per document/course and route by namespace.
- Add hybrid retrieval (keyword + vector) and optional reranking.
- Add deterministic mock mode for test and CI runs.

### Larger scale
- Replace local artifacts with object storage + managed vector store.
- Move session state and checkpoints to managed DB.
- Introduce queue-based ingestion and horizontal API workers.
- Add tracing and evaluation dashboards for retrieval quality and safety rates.

## Security and Operational Notes
- Do not store API keys in source.
- Keep provider credentials in environment/secret manager.
- Log request IDs and high-level metrics; avoid sensitive payload logging.
- Validate and constrain uploads and response payload sizes.
