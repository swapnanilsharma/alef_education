"""LLM-based answer synthesis using AWS Bedrock Claude."""

from __future__ import annotations

import logging

from app.core.schemas import AskMatch
from app.retrieval.bedrock_embeddings import BedrockLLMService

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a helpful tutor. Answer the student's question clearly and concisely "
    "using only the information provided in the context below. "
    "If the context does not contain enough information to answer, say so honestly. "
    "Do not invent citations or facts."
)


def synthesize_answer(
    question: str,
    grade_level: str,
    matches: list[AskMatch],
    recent_turns: list[dict] | None = None,
) -> str:
    """Generate an answer from retrieved chunks using Claude on Bedrock.

    Args:
        question: The student's question.
        grade_level: Student grade level used to tune answer style.
        matches: Top-k retrieved chunks from the FAISS index.
        recent_turns: Recent conversation turns for this student session.

    Returns:
        str: Generated answer from the LLM.
    """
    context = "\n\n".join(
        f"[Chunk {i + 1} | Page {m.page_number}]\n{m.text.strip()}"
        for i, m in enumerate(matches)
    )
    history_lines: list[str] = []
    for turn in recent_turns or []:
        history_lines.append(
            f"Q: {turn.get('question', '')}\n"
            f"A: {turn.get('answer', '')}\n"
            f"Status: {turn.get('safety_status', '')}"
        )
    history = "\n\n".join(history_lines) if history_lines else "No prior session context."

    prompt = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"Student level: {grade_level}. Use age-appropriate language and examples.\n\n"
        f"Recent session history (if helpful):\n{history}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    llm = BedrockLLMService()
    logger.info("Sending question to LLM | model_id=%s | chunks=%s", llm.model_id, len(matches))
    return llm.generate(prompt)
