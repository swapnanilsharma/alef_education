"""Question-answering endpoint."""

from __future__ import annotations

import logging
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from app.qa.qa_service import answer_question
from app.core.schemas import AskRequest, AskResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest) -> AskResponse:
    """Answer a question using the local FAISS vectorstore.

    Args:
        request: Question payload with student_id, question, and grade_level.

    Returns:
        AskResponse: Extractive answer with request_id and safety_status.

    Raises:
        HTTPException: If vectorstore artifacts are missing or question answering fails.
    """
    request_id = str(uuid4())
    logger.info(
        "Ask request received | request_id=%s | student_id=%s | grade_level=%s",
        request_id,
        request.student_id,
        request.grade_level,
    )

    try:
        response = answer_question(request, request_id)
    except FileNotFoundError as exc:
        logger.warning(
            "Ask request missing vectorstore artifacts | request_id=%s | reason=%s",
            request_id,
            exc,
        )
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        logger.warning(
            "Ask request rejected | request_id=%s | reason=%s", request_id, exc
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - convert retrieval/runtime issues into API error responses
        logger.exception("Ask request failed | request_id=%s", request_id)
        raise HTTPException(
            status_code=502, detail=f"Failed to answer question: {exc}"
        ) from exc

    logger.info("Ask request completed | request_id=%s", request_id)
    return response
