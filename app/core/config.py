"""Central configuration for the PDF ingestion and QA service."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
OUTPUTS_DIR = Path("outputs")
VECTOR_STORE_DIR = Path("vectorstores")
SESSION_DB_PATH = OUTPUTS_DIR / "qa_sessions.sqlite3"

# ---------------------------------------------------------------------------
# Bedrock — Embedding model
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"

# ---------------------------------------------------------------------------
# Bedrock — LLM (Amazon Nova Lite — no use-case form required)
# ---------------------------------------------------------------------------
LLM_MODEL_ID = "eu.amazon.nova-lite-v1:0"
LLM_MAX_TOKENS = 512

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_TARGET_CHARS = 1200
CHUNK_MAX_CHARS = 1600
CHUNK_OVERLAP = 120

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
RETRIEVAL_TOP_K = 5
MIN_RETRIEVAL_SCORE = 0.15

# ---------------------------------------------------------------------------
# Safety and answer constraints
# ---------------------------------------------------------------------------
SAFETY_STATUS_ANSWERED = "answered"
SAFETY_STATUS_BLOCKED = "blocked"
SAFETY_STATUS_OFF_TOPIC = "off_topic"
SAFETY_STATUS_INSUFFICIENT = "insufficient"
MAX_SOURCE_EXCERPT_CHARS = 240

# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------
SESSION_HISTORY_LIMIT = 5
