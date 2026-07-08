"""Question-answering service over local FAISS retrieval results."""

from __future__ import annotations

import logging

from app.core.schemas import AskRequest, AskResponse
from app.qa.graph import run_qa_graph

logger = logging.getLogger(__name__)


def answer_question(request: AskRequest, request_id: str) -> AskResponse:
    """Answer a question using the LangGraph QA workflow.

    Args:
        request: Question request containing student_id, question, and grade_level.
        request_id: Unique identifier for the request, used in the response.

    Returns:
        AskResponse: Answer payload with sources and safety status.
    """
    logger.info(
        "Answer request received | request_id=%s | student_id=%s | grade_level=%s",
        request_id,
        request.student_id,
        request.grade_level,
    )
    return run_qa_graph(request, request_id)


