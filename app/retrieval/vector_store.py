"""Local FAISS vector store persistence utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import faiss
import numpy as np

from app.core.schemas import AskMatch, EmbeddingChunk
from app.core.config import VECTOR_STORE_DIR
from app.ingestion.storage import build_artifact_stem, load_json_file, save_json_payload

logger = logging.getLogger(__name__)


def save_faiss_index(
    chunks: list[EmbeddingChunk],
    embeddings: list[list[float]],
    output_dir: Path,
    base_name: str,
    source_json_path: Path,
    model_id: str,
) -> tuple[Path, Path]:
    """Persist normalized embeddings to a local FAISS index and metadata JSON.

    Args:
        chunks: Chunk metadata aligned with the embeddings.
        embeddings: Embedding vectors in the same order as the chunk list.
        output_dir: Directory where vector artifacts are stored.
        base_name: Base artifact name.
        source_json_path: JSON document used to build the embeddings.
        model_id: Bedrock model identifier used during embedding creation.

    Returns:
        tuple[Path, Path]: Paths to the FAISS index and metadata JSON files.

    Raises:
        ValueError: If no embeddings are provided.
    """
    if not embeddings:
        raise ValueError("No embeddings were provided for FAISS persistence.")

    logger.info(
        "Persisting FAISS index | output_dir=%s | base_name=%s | chunk_count=%s",
        output_dir,
        base_name,
        len(chunks),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    vector_array = np.ascontiguousarray(np.asarray(embeddings, dtype="float32"))
    faiss.normalize_L2(x=vector_array)

    index = faiss.IndexFlatIP(vector_array.shape[1])
    index.add(vector_array)

    artifact_stem = build_artifact_stem(output_dir, base_name)
    index_path = artifact_stem.with_suffix(".faiss")
    metadata_path = artifact_stem.with_suffix(".metadata.json")

    faiss.write_index(index, str(index_path))
    save_json_payload(
        {
            "source_json": str(source_json_path),
            "model_id": model_id,
            "metric": "inner_product",
            "normalized": True,
            "dimension": int(vector_array.shape[1]),
            "chunk_count": len(chunks),
            "chunks": [chunk.model_dump() for chunk in chunks],
        },
        metadata_path,
    )

    logger.info(
        "FAISS artifacts saved | index_path=%s | metadata_path=%s | dimension=%s | chunk_count=%s",
        index_path,
        metadata_path,
        vector_array.shape[1],
        len(chunks),
    )

    return index_path, metadata_path


def resolve_vectorstore_paths(
    index_path: str | None = None,
    metadata_path: str | None = None,
) -> tuple[Path, Path]:
    """Resolve vectorstore artifact paths, defaulting to the latest saved index.

    Args:
        index_path: Optional explicit FAISS index path.
        metadata_path: Optional explicit metadata JSON path.

    Returns:
        tuple[Path, Path]: Resolved FAISS index path and metadata path.

    Raises:
        FileNotFoundError: If the required artifacts cannot be found.
    """
    if index_path:
        resolved_index_path = Path(index_path)
    else:
        candidates = sorted(VECTOR_STORE_DIR.glob("*.faiss"), key=lambda path: path.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError("No FAISS index found in vectorstores directory.")
        resolved_index_path = candidates[-1]

    if not resolved_index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {resolved_index_path}")

    if metadata_path:
        resolved_metadata_path = Path(metadata_path)
    else:
        resolved_metadata_path = resolved_index_path.with_suffix(".metadata.json")

    if not resolved_metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {resolved_metadata_path}")

    return resolved_index_path, resolved_metadata_path


def search_faiss_index(
    index_path: Path,
    metadata_path: Path,
    query_embedding: list[float],
    top_k: int,
) -> tuple[list[AskMatch], dict]:
    """Search a local FAISS index using a normalized query embedding.

    Args:
        index_path: Path to the FAISS index file.
        metadata_path: Path to the companion metadata JSON file.
        query_embedding: Embedding vector for the question.
        top_k: Number of matches to return.

    Returns:
        tuple[list[AskMatch], dict]: Search matches and the raw metadata payload.
    """
    logger.info(
        "Searching FAISS index | index_path=%s | metadata_path=%s | top_k=%s",
        index_path,
        metadata_path,
        top_k,
    )
    metadata = load_json_file(metadata_path)
    chunks = metadata.get("chunks", [])

    index = faiss.read_index(str(index_path))
    query_array = np.ascontiguousarray(np.asarray([query_embedding], dtype="float32"))
    if metadata.get("normalized", False):
        faiss.normalize_L2(x=query_array)

    limit = min(top_k, len(chunks))
    distances, labels = index.search(query_array, limit)

    matches: list[AskMatch] = []
    for score, label in zip(distances[0], labels[0]):
        if label < 0 or label >= len(chunks):
            continue

        chunk = chunks[int(label)]
        matches.append(
            AskMatch(
                chunk_id=chunk["chunk_id"],
                page_number=chunk["page_number"],
                section_heading=chunk.get("section_heading", "Unknown section"),
                score=float(score),
                text=chunk["text"],
            )
        )

    logger.info("FAISS search completed | index_path=%s | match_count=%s", index_path, len(matches))
    return matches, metadata
