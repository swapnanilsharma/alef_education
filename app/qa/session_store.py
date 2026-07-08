"""Local SQLite storage for QA sessions and checkpoints."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from app.core.config import SESSION_DB_PATH


def _connect() -> sqlite3.Connection:
    """Create a SQLite connection configured with row-dict style access."""
    SESSION_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(SESSION_DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_session_store() -> None:
    """Create required session, turn, and checkpoint tables when absent."""
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                student_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                request_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                safety_status TEXT NOT NULL,
                top_score REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                request_id TEXT NOT NULL,
                state_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def get_or_create_thread_id(student_id: str) -> str:
    """Return a stable thread id for a student, creating one if necessary.

    Args:
        student_id: Unique identifier for the student.

    Returns:
        str: Persistent thread id used for LangGraph execution config.
    """
    now = datetime.now(timezone.utc).isoformat()
    default_thread_id = f"student:{student_id}"
    with _connect() as connection:
        row = connection.execute(
            "SELECT thread_id FROM sessions WHERE student_id = ?",
            (student_id,),
        ).fetchone()
        if row:
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE student_id = ?",
                (now, student_id),
            )
            return str(row["thread_id"])

        connection.execute(
            "INSERT INTO sessions (student_id, thread_id, updated_at) VALUES (?, ?, ?)",
            (student_id, default_thread_id, now),
        )
        return default_thread_id


def get_recent_turns(student_id: str, limit: int) -> list[dict[str, Any]]:
    """Fetch recent conversation turns for a student in chronological order.

    Args:
        student_id: Unique identifier for the student.
        limit: Maximum number of turns to return.

    Returns:
        list[dict[str, Any]]: Turn payloads with question/answer metadata.
    """
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT question, answer, safety_status, top_score, created_at
            FROM turns
            WHERE student_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (student_id, limit),
        ).fetchall()
    # Return oldest->newest for prompt coherence.
    turns = [dict(row) for row in rows]
    turns.reverse()
    return turns


def save_turn(
    student_id: str,
    request_id: str,
    question: str,
    answer: str,
    safety_status: str,
    top_score: float,
) -> None:
    """Persist one completed ask turn for session recall and analysis.

    Args:
        student_id: Unique identifier for the student.
        request_id: API request id for traceability.
        question: Student question text.
        answer: Final answer text returned by the assistant.
        safety_status: Final safety routing result.
        top_score: Highest retrieval similarity score for the turn.
    """
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO turns (student_id, request_id, question, answer, safety_status, top_score, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (student_id, request_id, question, answer, safety_status, top_score, now),
        )


def save_checkpoint(thread_id: str, request_id: str, state: dict[str, Any]) -> None:
    """Persist serialized graph state as a local SQLite checkpoint.

    Args:
        thread_id: Session thread id used for graph execution.
        request_id: API request id for traceability.
        state: JSON-serializable snapshot of the final graph state.
    """
    now = datetime.now(timezone.utc).isoformat()
    serialized = json.dumps(state, ensure_ascii=True)
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO checkpoints (thread_id, request_id, state_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (thread_id, request_id, serialized, now),
        )
