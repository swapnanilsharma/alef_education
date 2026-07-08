"""PDF ingestion endpoint."""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import uuid4
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.ingestion.embedding_pipeline import build_embeddings_from_extracted_pdf
from app.ingestion.pdf_extractor import extract_pdf_pagewise_from_bytes
from app.core.schemas import IngestResponse
from app.core.config import OUTPUTS_DIR
from app.ingestion.storage import build_output_path, save_to_json

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)) -> IngestResponse:
    """Ingest an uploaded PDF, persist cleaned JSON, and build local embeddings.

    Args:
        file: Multipart upload payload containing the PDF document.

    Returns:
        IngestResponse: API response with JSON and FAISS artifact details.

    Raises:
        HTTPException: If the upload is not a PDF, is empty, cannot be parsed, or embedding build fails.
    """
    request_id = str(uuid4())
    logger.info(
        "Ingest request received | request_id=%s | filename=%s | content_type=%s",
        request_id,
        file.filename,
        file.content_type,
    )

    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        logger.warning(
            "Rejected upload with unsupported content type | request_id=%s | content_type=%s",
            request_id,
            file.content_type,
        )
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        logger.warning("Rejected empty upload | request_id=%s", request_id)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    filename = file.filename or "uploaded_pdf"
    stem = Path(filename).stem or "uploaded_pdf"
    logger.info(
        "Upload accepted | request_id=%s | filename=%s | size_bytes=%s",
        request_id,
        filename,
        len(pdf_bytes),
    )

    try:
        extracted = extract_pdf_pagewise_from_bytes(pdf_bytes, source_name=filename)
    except Exception as exc:  # noqa: BLE001 - return clear client error for invalid PDF payloads
        logger.exception("PDF extraction failed | request_id=%s | filename=%s", request_id, filename)
        raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {exc}") from exc

    output_path = build_output_path(OUTPUTS_DIR, stem)
    save_to_json(extracted, output_path)
    logger.info(
        "Extracted JSON saved | request_id=%s | output_json=%s | total_pages=%s",
        request_id,
        output_path,
        extracted.total_pages,
    )

    try:
        index_path, metadata_path, chunk_count, model_id = build_embeddings_from_extracted_pdf(
            extracted_pdf=extracted,
            source_json_path=output_path,
            index_name=stem,
        )
    except ValueError as exc:
        logger.warning("Embedding build rejected | request_id=%s | reason=%s", request_id, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - convert Bedrock/runtime issues into API error responses
        logger.exception("Embedding build failed | request_id=%s | source_json=%s", request_id, output_path)
        raise HTTPException(status_code=502, detail=f"Failed to build embeddings: {exc}") from exc

    logger.info(
        "Ingest completed | request_id=%s | index_path=%s | metadata_path=%s | chunk_count=%s | model_id=%s",
        request_id,
        index_path,
        metadata_path,
        chunk_count,
        model_id,
    )
    return IngestResponse(
        message="PDF ingested and embeddings built successfully.",
        output_json=str(output_path),
        total_pages=extracted.total_pages,
        index_path=str(index_path),
        metadata_path=str(metadata_path),
        chunk_count=chunk_count,
        model_id=model_id,
    )
