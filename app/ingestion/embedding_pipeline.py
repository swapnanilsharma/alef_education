"""Embedding pipeline that transforms cleaned JSON into local FAISS artifacts."""

from __future__ import annotations

import logging
from pathlib import Path
from app.retrieval.bedrock_embeddings import BedrockEmbeddingService
from app.core.config import VECTOR_STORE_DIR
from app.ingestion.embedding_chunker import build_embedding_chunks
from app.core.schemas import ExtractedPdf
from app.retrieval.vector_store import save_faiss_index
logger = logging.getLogger(__name__)


def build_embeddings_from_extracted_pdf(
    extracted_pdf: ExtractedPdf,
    source_json_path: Path,
    index_name: str | None = None,
    embedding_service: BedrockEmbeddingService | None = None,
) -> tuple[Path, Path, int, str]:
    """Create local FAISS artifacts from an in-memory extracted PDF payload.

    Args:
        extracted_pdf: Extracted PDF payload containing cleaned page text.
        source_json_path: JSON path associated with the extracted payload.
        index_name: Optional custom artifact base name.
        embedding_service: Optional preconfigured Bedrock embedding client.

    Returns:
        tuple[Path, Path, int, str]: Index path, metadata path, chunk count, and model id.

    Raises:
        ValueError: If the extracted PDF contains no clean text chunks.
    """
    logger.info(
        "Starting embedding build from extracted PDF | source_json=%s | total_pages=%s | index_name=%s",
        source_json_path,
        extracted_pdf.total_pages,
        index_name,
    )
    chunks = build_embedding_chunks(extracted_pdf)
    if not chunks:
        raise ValueError("No cleaned text was available to create embeddings.")

    logger.info("Chunking completed | source_json=%s | chunk_count=%s", source_json_path, len(chunks))

    embedder = embedding_service or BedrockEmbeddingService()
    logger.info(
        "Requesting embeddings from Bedrock | source_json=%s | model_id=%s | chunk_count=%s",
        source_json_path,
        embedder.model_id,
        len(chunks),
    )
    embeddings = embedder.embed_texts([chunk.text for chunk in chunks])

    index_base_name = index_name or source_json_path.stem
    index_path, metadata_path = save_faiss_index(
        chunks=chunks,
        embeddings=embeddings,
        output_dir=VECTOR_STORE_DIR,
        base_name=index_base_name,
        source_json_path=source_json_path,
        model_id=embedder.model_id,
    )
    logger.info(
        "Embedding build completed | source_json=%s | index_path=%s | metadata_path=%s | model_id=%s",
        source_json_path,
        index_path,
        metadata_path,
        embedder.model_id,
    )
    return index_path, metadata_path, len(chunks), embedder.model_id
