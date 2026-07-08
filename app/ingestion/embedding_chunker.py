"""Chunk cleaned PDF text into embedding-friendly segments."""

from __future__ import annotations

from app.core.config import CHUNK_MAX_CHARS, CHUNK_OVERLAP
from app.core.schemas import EmbeddingChunk, ExtractedPdf


def build_embedding_chunks(
    extracted_pdf: ExtractedPdf,
    max_chars: int = CHUNK_MAX_CHARS,
    overlap: int = CHUNK_OVERLAP,
) -> list[EmbeddingChunk]:
    """Split cleaned page text into bounded chunks for embedding generation.

    Args:
        extracted_pdf: Extracted PDF payload containing cleaned page text.
        max_chars: Maximum number of characters per chunk.
        overlap: Character overlap between consecutive chunks.

    Returns:
        list[EmbeddingChunk]: Ordered chunks ready to send to the embedding model.
    """
    chunks: list[EmbeddingChunk] = []

    for page in extracted_pdf.pages:
        text = page.text.strip()
        if not text:
            continue
        section_heading = _infer_section_heading(text)

        start = 0
        chunk_number = 1
        text_length = len(text)

        while start < text_length:
            end = min(start + max_chars, text_length)
            if end < text_length:
                split_end = _find_split_boundary(text, start, end, minimum_split=start + (max_chars // 2))
                if split_end is not None:
                    end = split_end

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    EmbeddingChunk(
                        chunk_id=f"page-{page.page_number}-chunk-{chunk_number}",
                        page_number=page.page_number,
                        section_heading=section_heading,
                        text=chunk_text,
                        start_offset=start,
                        end_offset=end,
                    )
                )
                chunk_number += 1

            if end >= text_length:
                break

            next_start = max(end - overlap, start + 1)
            if next_start <= start:
                next_start = end
            start = next_start

    return chunks


def _find_split_boundary(text: str, start: int, end: int, minimum_split: int) -> int | None:
    """Find a natural boundary for chunk splitting.

    Args:
        text: Full cleaned page text.
        start: Current chunk start offset.
        end: Current chunk end offset.
        minimum_split: Minimum acceptable offset for a split candidate.

    Returns:
        int | None: Preferred split offset if found, otherwise None.
    """
    del start

    for separator in ("\n\n", "\n", " "):
        candidate = text.rfind(separator, minimum_split, end)
        if candidate != -1:
            return candidate + len(separator)

    return None


def _infer_section_heading(page_text: str) -> str:
    """Infer a section heading from the first few non-empty page lines."""
    lines = [line.strip() for line in page_text.split("\n") if line.strip()]
    for line in lines[:8]:
        if len(line) > 90:
            continue
        if line.endswith(":"):
            return line.rstrip(":")
        words = line.split()
        if 1 <= len(words) <= 8 and line == line.title():
            return line
        if 1 <= len(words) <= 8 and line.isupper():
            return line.title()
    return "Unknown section"
