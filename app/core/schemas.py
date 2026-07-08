"""Pydantic models used by the PDF ingestion service."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PageData(BaseModel):
    """Structured text extracted from a single PDF page."""

    page_number: int = Field(description="1-based page number in the source PDF.")
    raw_text: str = Field(description="Raw text extracted directly from the PDF page.")
    text: str = Field(description="Noise-filtered text optimized for downstream embeddings.")


class ExtractedPdf(BaseModel):
    """Complete page-wise extraction payload for an ingested PDF."""

    source_pdf: str
    total_pages: int
    pages: list[PageData]


class IngestResponse(BaseModel):
    """API response returned after successful PDF ingestion and embedding build."""

    message: str
    output_json: str
    total_pages: int
    index_path: str
    metadata_path: str
    chunk_count: int
    model_id: str


class ChunkSourceSpan(BaseModel):
    """Source span contributing text from a single page into a chunk."""

    page_number: int = Field(description="1-based page number in the source PDF.")
    start_offset: int = Field(description="Start character offset in the cleaned page text.")
    end_offset: int = Field(description="End character offset in the cleaned page text.")


class EmbeddingChunk(BaseModel):
    """Chunk of cleaned PDF text prepared for embedding and retrieval."""

    chunk_id: str = Field(description="Unique identifier for the embedding chunk.")
    page_start: int = Field(description="First page number contributing text to the chunk.")
    page_end: int = Field(description="Last page number contributing text to the chunk.")
    section_heading: str = Field(description="Best-effort section heading inferred for the chunk.")
    chunk_kind: str = Field(description="Primary semantic kind of the chunk, such as paragraph or practice.")
    text: str = Field(description="Clean text content used to generate the embedding.")
    source_spans: list[ChunkSourceSpan] = Field(description="Per-page source spans contributing to the chunk.")


class AskRequest(BaseModel):
    """Student question payload accepted by the ask API."""

    student_id: str = Field(description="Unique identifier for the student.")
    question: str = Field(description="Natural-language question to ask against the local vectorstore.")
    grade_level: str = Field(description="Grade level of the student (e.g., 'grade_8').")


class AskMatch(BaseModel):
    """One retrieval hit returned from the vector search layer."""

    chunk_id: str
    page_start: int
    page_end: int
    section_heading: str
    chunk_kind: str
    score: float
    text: str


class AskSource(BaseModel):
    """Source reference returned to the API client for answer attribution."""

    source_id: str
    title: str
    page: int = Field(description="Starting page number; equals page_start. Preserved for backward compatibility.")
    page_start: int = Field(description="First page number contributing text to this source chunk.")
    page_end: int = Field(description="Last page number contributing text to this source chunk.")
    excerpt: str


class AskResponse(BaseModel):
    """Final answer payload returned by the ask API."""

    request_id: str
    answer: str
    sources: list[AskSource]
    safety_status: str
