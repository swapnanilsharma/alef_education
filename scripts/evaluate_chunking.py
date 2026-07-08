"""Evaluate semantic chunking and source metadata quality on extracted PDF JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.schemas import ExtractedPdf
from app.ingestion.embedding_chunker import build_embedding_chunks


def evaluate(extracted_json_path: Path, show_examples: int) -> int:
    """Evaluate chunking quality and print summary statistics."""
    if not extracted_json_path.exists():
        print(f"ERROR: file not found: {extracted_json_path}")
        return 1

    with extracted_json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    extracted = ExtractedPdf.model_validate(payload)
    chunks = build_embedding_chunks(extracted)

    if not chunks:
        print("ERROR: no chunks produced")
        return 2

    invalid_ranges = [c.chunk_id for c in chunks if c.page_start > c.page_end]
    missing_spans = [c.chunk_id for c in chunks if not c.source_spans]
    empty_text = [c.chunk_id for c in chunks if not c.text.strip()]

    cross_page_count = sum(1 for c in chunks if c.page_start != c.page_end)
    avg_chars = int(sum(len(c.text) for c in chunks) / len(chunks))

    print("=== Chunking Evaluation ===")
    print(f"source_pdf: {extracted.source_pdf}")
    print(f"total_pages: {extracted.total_pages}")
    print(f"chunk_count: {len(chunks)}")
    print(f"cross_page_chunks: {cross_page_count}")
    print(f"avg_chunk_chars: {avg_chars}")
    print(f"invalid_page_ranges: {len(invalid_ranges)}")
    print(f"missing_source_spans: {len(missing_spans)}")
    print(f"empty_chunks: {len(empty_text)}")

    print("\n=== Sample Chunks ===")
    for chunk in chunks[: max(0, show_examples)]:
        print(
            f"{chunk.chunk_id} | pages {chunk.page_start}-{chunk.page_end} | "
            f"kind={chunk.chunk_kind} | section={chunk.section_heading} | chars={len(chunk.text)}"
        )

    if invalid_ranges or missing_spans or empty_text:
        print("\nFAIL: quality checks failed")
        return 3

    print("\nPASS: chunking checks passed")
    return 0


def main() -> int:
    """Command-line entry point for chunking evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate chunking output for extracted PDF JSON"
    )
    parser.add_argument(
        "extracted_json",
        type=Path,
        help="Path to extracted JSON payload (for example outputs/<file>.json)",
    )
    parser.add_argument(
        "--show-examples", type=int, default=10, help="Number of sample chunks to print"
    )
    args = parser.parse_args()
    return evaluate(args.extracted_json, args.show_examples)


if __name__ == "__main__":
    raise SystemExit(main())
