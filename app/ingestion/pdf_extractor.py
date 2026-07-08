"""PDF extraction service powered by PyMuPDF."""

from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF

from app.core.schemas import ExtractedPdf, PageData
from app.ingestion.text_cleaner import clean_page_text

logger = logging.getLogger(__name__)


def extract_pdf_pagewise_from_bytes(
    pdf_bytes: bytes,
    source_name: str,
) -> ExtractedPdf:
    """Extract page-wise text from an in-memory PDF payload.

    Args:
        pdf_bytes: Raw bytes of the uploaded PDF.
        source_name: Name used in the output metadata to identify the source.

    Returns:
        ExtractedPdf: Structured page-wise extraction result.
    """
    logger.info(
        "Starting PDF extraction from bytes | source_name=%s | size_bytes=%s",
        source_name,
        len(pdf_bytes),
    )
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        pages: list[PageData] = []
        logger.debug(
            "Opened PDF document | source_name=%s | total_pages=%s",
            source_name,
            len(document),
        )
        for page_index, page in enumerate(document, start=1):
            raw_text = page.get_text("text").strip()
            cleaned_text = clean_page_text(raw_text)
            logger.debug(
                "Processed page | source_name=%s | page_number=%s | raw_chars=%s | cleaned_chars=%s",
                source_name,
                page_index,
                len(raw_text),
                len(cleaned_text),
            )
            pages.append(
                PageData(page_number=page_index, raw_text=raw_text, text=cleaned_text)
            )

    logger.info(
        "Completed PDF extraction | source_name=%s | total_pages=%s",
        source_name,
        len(pages),
    )
    return ExtractedPdf(source_pdf=source_name, total_pages=len(pages), pages=pages)


def extract_pdf_pagewise_from_path(
    pdf_path: Path,
) -> ExtractedPdf:
    """Extract page-wise text from a PDF file on disk.

    Args:
        pdf_path: Filesystem path of the PDF to read.

    Returns:
        ExtractedPdf: Structured page-wise extraction result.

    Raises:
        FileNotFoundError: If the provided path does not exist.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("Starting PDF extraction from path | pdf_path=%s", pdf_path)
    with pdf_path.open("rb") as file:
        return extract_pdf_pagewise_from_bytes(file.read(), source_name=str(pdf_path))
