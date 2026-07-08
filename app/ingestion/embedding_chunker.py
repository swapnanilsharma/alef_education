"""Chunk cleaned PDF text into embedding-friendly segments."""

from __future__ import annotations

from dataclasses import dataclass
import re

from app.core.config import CHUNK_MAX_CHARS, CHUNK_OVERLAP, CHUNK_TARGET_CHARS
from app.core.schemas import ChunkSourceSpan, EmbeddingChunk, ExtractedPdf

_EXAMPLE_PREFIXES = ("for example", "example", "let’s try", "let's try", "solve:", "simplify:")
_PRACTICE_PREFIXES = ("your turn", "now you try", "practice")
_GENERIC_NON_SECTION_HEADINGS = {
    "equation",
    "expression",
    "simplify",
    "solve",
    "check",
    "length",
    "width",
    "yard",
}


@dataclass(slots=True)
class _SemanticBlock:
    page_number: int
    start_offset: int
    end_offset: int
    text: str
    block_kind: str
    section_heading: str = "Unknown section"


def build_embedding_chunks(
    extracted_pdf: ExtractedPdf,
    target_chars: int = CHUNK_TARGET_CHARS,
    max_chars: int = CHUNK_MAX_CHARS,
    overlap: int = CHUNK_OVERLAP,
) -> list[EmbeddingChunk]:
    """Split cleaned PDF text into semantic, page-attributed chunks.

    Args:
        extracted_pdf: Extracted PDF payload containing cleaned page text.
        target_chars: Preferred target size for each chunk.
        max_chars: Maximum number of characters per chunk.
        overlap: Preferred overlap size between consecutive chunks.

    Returns:
        list[EmbeddingChunk]: Ordered chunks ready to send to the embedding model.
    """
    blocks = _build_semantic_blocks(extracted_pdf)
    if not blocks:
        return []

    return _pack_blocks_into_chunks(blocks, target_chars=target_chars, max_chars=max_chars, overlap=overlap)


def _build_semantic_blocks(extracted_pdf: ExtractedPdf) -> list[_SemanticBlock]:
    """Build semantic blocks from page text while preserving page-local offsets."""
    blocks: list[_SemanticBlock] = []
    active_heading = "Unknown section"

    for page in extracted_pdf.pages:
        page_text = page.text.strip()
        if not page_text:
            continue

        page_blocks = _split_page_into_blocks(page.page_number, page.text)
        for block in page_blocks:
            if block.block_kind == "heading" and _is_section_heading(block.text):
                active_heading = _normalize_heading(block.text)
                block.section_heading = active_heading
            else:
                block.section_heading = active_heading
            blocks.append(block)

    return blocks


def _split_page_into_blocks(page_number: int, text: str) -> list[_SemanticBlock]:
    """Split a page into semantic blocks using blank lines and cue-based boundaries."""
    blocks: list[_SemanticBlock] = []
    current_lines: list[str] = []
    current_start: int | None = None
    line_start = 0

    for raw_line in text.splitlines(keepends=True):
        stripped_line = raw_line.strip()

        if not stripped_line:
            _flush_block(blocks, page_number, current_lines, current_start, line_start)
            current_lines = []
            current_start = None
            line_start += len(raw_line)
            continue

        if current_lines and _starts_strong_boundary(stripped_line):
            _flush_block(blocks, page_number, current_lines, current_start, line_start)
            current_lines = []
            current_start = None

        if current_start is None:
            current_start = line_start

        current_lines.append(stripped_line)
        line_start += len(raw_line)

    _flush_block(blocks, page_number, current_lines, current_start, len(text))
    return blocks


def _flush_block(
    blocks: list[_SemanticBlock],
    page_number: int,
    current_lines: list[str],
    current_start: int | None,
    current_end: int,
) -> None:
    """Finalize and append a semantic block when buffered lines are present."""
    if not current_lines or current_start is None:
        return

    block_text = "\n".join(current_lines).strip()
    if not block_text:
        return

    blocks.append(
        _SemanticBlock(
            page_number=page_number,
            start_offset=current_start,
            end_offset=current_end,
            text=block_text,
            block_kind=_classify_block(current_lines),
        )
    )


def _pack_blocks_into_chunks(
    blocks: list[_SemanticBlock],
    target_chars: int,
    max_chars: int,
    overlap: int,
) -> list[EmbeddingChunk]:
    """Pack semantic blocks into bounded embedding chunks."""
    chunks: list[EmbeddingChunk] = []
    current_blocks: list[_SemanticBlock] = []
    current_chars = 0

    for block in blocks:
        if current_blocks and _should_flush_before_block(current_blocks, block, current_chars, target_chars, max_chars):
            chunks.append(_make_chunk(len(chunks) + 1, current_blocks))
            current_blocks = _select_overlap_blocks(current_blocks, overlap)
            current_chars = _blocks_char_count(current_blocks)

        current_blocks.append(block)
        current_chars += len(block.text)

        if current_chars >= max_chars:
            chunks.append(_make_chunk(len(chunks) + 1, current_blocks))
            current_blocks = _select_overlap_blocks(current_blocks, overlap)
            current_chars = _blocks_char_count(current_blocks)

    if current_blocks:
        chunks.append(_make_chunk(len(chunks) + 1, current_blocks))

    return chunks


def _should_flush_before_block(
    current_blocks: list[_SemanticBlock],
    next_block: _SemanticBlock,
    current_chars: int,
    target_chars: int,
    max_chars: int,
) -> bool:
    """Decide whether the current chunk should be flushed before appending a block."""
    if current_chars + len(next_block.text) > max_chars:
        return True

    if current_chars < max(300, target_chars // 2):
        return False

    if next_block.block_kind == "heading":
        return True

    if next_block.section_heading != current_blocks[-1].section_heading and current_chars >= target_chars:
        return True

    if next_block.block_kind in {"worked_example", "practice"} and current_chars >= target_chars:
        return True

    return current_chars >= target_chars and current_blocks[-1].block_kind == "equation_group"


def _select_overlap_blocks(blocks: list[_SemanticBlock], overlap: int) -> list[_SemanticBlock]:
    """Carry a small trailing semantic overlap into the next chunk."""
    if overlap <= 0 or not blocks:
        return []

    kept: list[_SemanticBlock] = []
    total = 0
    for block in reversed(blocks):
        if block.block_kind == "heading" and kept:
            continue
        kept.insert(0, block)
        total += len(block.text)
        if total >= overlap:
            break
    return kept


def _make_chunk(chunk_number: int, blocks: list[_SemanticBlock]) -> EmbeddingChunk:
    """Create an embedding chunk from a list of semantic blocks."""
    page_start = min(block.page_number for block in blocks)
    page_end = max(block.page_number for block in blocks)
    section_heading = next(
        (block.section_heading for block in reversed(blocks) if block.section_heading != "Unknown section"),
        "Unknown section",
    )
    primary_block = next((block for block in blocks if block.block_kind != "heading"), blocks[0])
    source_spans = [
        ChunkSourceSpan(
            page_number=block.page_number,
            start_offset=block.start_offset,
            end_offset=block.end_offset,
        )
        for block in blocks
    ]

    return EmbeddingChunk(
        chunk_id=f"chunk-{chunk_number}",
        page_start=page_start,
        page_end=page_end,
        section_heading=section_heading,
        chunk_kind=primary_block.block_kind,
        text="\n\n".join(block.text for block in blocks).strip(),
        source_spans=source_spans,
    )


def _blocks_char_count(blocks: list[_SemanticBlock]) -> int:
    """Return the total character count of a list of blocks."""
    return sum(len(block.text) for block in blocks)


def _classify_block(lines: list[str]) -> str:
    """Classify a buffered block into a coarse semantic type."""
    first_line = lines[0].strip()
    lowered_first = first_line.lower()

    if _is_section_heading(first_line):
        return "heading"
    if lowered_first.startswith(_PRACTICE_PREFIXES):
        return "practice"
    if lowered_first.startswith(_EXAMPLE_PREFIXES):
        return "worked_example"
    if _looks_like_list_block(lines):
        return "list"
    if _looks_like_equation_group(lines):
        return "equation_group"
    return "paragraph"


def _starts_strong_boundary(line: str) -> bool:
    """Return whether a line strongly indicates a new semantic block."""
    return _is_section_heading(line) or _has_prefix(line, _EXAMPLE_PREFIXES) or _has_prefix(line, _PRACTICE_PREFIXES)


def _looks_like_equation_group(lines: list[str]) -> bool:
    """Heuristically detect math-heavy blocks that should stay together."""
    equation_like = 0
    for line in lines:
        if re.search(r"[=<>±÷×]|\b\d+[a-zA-Z]?\b", line) and len(line.split()) <= 12:
            equation_like += 1
    return equation_like >= max(2, len(lines) // 2)


def _looks_like_list_block(lines: list[str]) -> bool:
    """Detect simple numbered or bulleted lists."""
    return sum(1 for line in lines if re.match(r"^(\d+\.|[-*])\s+", line)) >= 2


def _is_section_heading(text: str) -> bool:
    """Infer whether a line is a likely section heading."""
    line = text.strip().rstrip(":")
    if not line or len(line) > 80:
        return False

    lowered = line.lower()
    if _has_prefix(line, _EXAMPLE_PREFIXES) or _has_prefix(line, _PRACTICE_PREFIXES):
        return False

    if re.match(r"^\d+\.", line):
        return False

    if re.search(r"[=<>]", line):
        return False

    if not any(char.isalpha() for char in line):
        return False

    words = line.split()
    if not 1 <= len(words) <= 8:
        return False

    if len(words) == 1 and lowered in _GENERIC_NON_SECTION_HEADINGS:
        return False

    if line.isupper():
        return True

    return line == line.title()


def _normalize_heading(text: str) -> str:
    """Normalize a heading for chunk metadata."""
    return text.strip().rstrip(":")


def _has_prefix(text: str, prefixes: tuple[str, ...]) -> bool:
    """Check prefixes after normalizing surrounding whitespace and trailing colons."""
    normalized = text.strip().lower().rstrip(":")
    return normalized.startswith(tuple(prefix.rstrip(":") for prefix in prefixes))
