"""LangGraph-based QA workflow with safety, retrieval, confidence, and answer nodes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.core.config import (
    MAX_SOURCE_EXCERPT_CHARS,
    MIN_RETRIEVAL_SCORE,
    RETRIEVAL_TOP_K,
    SESSION_HISTORY_LIMIT,
    SAFETY_STATUS_ANSWERED,
    SAFETY_STATUS_BLOCKED,
    SAFETY_STATUS_INSUFFICIENT,
    SAFETY_STATUS_OFF_TOPIC,
)
from app.core.schemas import AskMatch, AskRequest, AskResponse, AskSource
from app.qa.answer_synthesizer import synthesize_answer
from app.qa.session_store import (
    get_or_create_thread_id,
    get_recent_turns,
    initialize_session_store,
    save_checkpoint,
    save_turn,
)
from app.retrieval.bedrock_embeddings import BedrockEmbeddingService, BedrockLLMService
from app.retrieval.vector_store import resolve_vectorstore_paths, search_faiss_index

logger = logging.getLogger(__name__)


class QAState(TypedDict, total=False):
    """Mutable state shared across LangGraph QA workflow nodes."""

    request: AskRequest
    request_id: str
    thread_id: str
    recent_turns: list[dict]
    safety_status: str
    answer: str
    sources: list[AskSource]
    matches: list[AskMatch]
    top_score: float


def _safety_node(state: QAState) -> QAState:
    """Classify the request as safe/off-topic/blocked before retrieval.

    Args:
        state: Current graph state containing the ask request.

    Returns:
        QAState: Partial state update with safety status and optional early answer.
    """
    request = state["request"]
    llm = BedrockLLMService()
    prompt = (
        "Classify the student request into exactly one label: safe, off_topic, or blocked. "
        "Use blocked for harmful or academic dishonesty requests (e.g., cheating, exam leakage). "
        "Use off_topic if unrelated to learning algebra from a textbook context. "
        "Return only the label.\n\n"
        f"Question: {request.question}"
    )
    label = llm.generate(prompt).strip().lower().split()[0]
    if label.startswith("blocked"):
        return {
            "safety_status": SAFETY_STATUS_BLOCKED,
            "answer": "I can't help with that request, but I can help you learn the concept step-by-step.",
            "sources": [],
        }
    if label.startswith("off_topic"):
        return {
            "safety_status": SAFETY_STATUS_OFF_TOPIC,
            "answer": "I can help with algebra questions from the provided book. Please ask a related question.",
            "sources": [],
        }
    return {"safety_status": SAFETY_STATUS_ANSWERED}


def _retrieve_node(state: QAState) -> QAState:
    """Retrieve top-k relevant chunks and build source references.

    Args:
        state: Current graph state containing the ask request.

    Returns:
        QAState: Partial state update with retrieval matches, sources, and top score.
    """
    request = state["request"]
    embedding_service = BedrockEmbeddingService()
    index_path, metadata_path = resolve_vectorstore_paths()
    query_embedding = embedding_service.embed_text(request.question)
    matches, metadata = search_faiss_index(
        index_path=index_path,
        metadata_path=metadata_path,
        query_embedding=query_embedding,
        top_k=RETRIEVAL_TOP_K,
    )
    sources = _build_sources(matches, metadata.get("source_json", ""))
    top_score = max((match.score for match in matches), default=0.0)
    return {"matches": matches, "sources": sources, "top_score": top_score}


def _confidence_node(state: QAState) -> QAState:
    """Gate answering when retrieval is empty or too low confidence.

    Args:
        state: Current graph state containing retrieval outputs.

    Returns:
        QAState: Partial state update with safety status and fallback answer when needed.
    """
    if not state.get("matches"):
        return {
            "safety_status": SAFETY_STATUS_INSUFFICIENT,
            "answer": "I could not find relevant content in the book to answer that.",
            "sources": [],
        }

    if state.get("top_score", 0.0) < MIN_RETRIEVAL_SCORE:
        return {
            "safety_status": SAFETY_STATUS_INSUFFICIENT,
            "answer": "The book does not seem to contain enough relevant information for that question.",
            "sources": state.get("sources", []),
        }

    return {"safety_status": state.get("safety_status", SAFETY_STATUS_ANSWERED)}


def _answer_node(state: QAState) -> QAState:
    """Generate the final answer using retrieved context and session history.

    Args:
        state: Current graph state containing request, matches, and recent turns.

    Returns:
        QAState: Partial state update with generated answer and answered status.
    """
    request = state["request"]
    answer = synthesize_answer(
        request.question,
        request.grade_level,
        state.get("matches", []),
        state.get("recent_turns", []),
    )
    return {
        "answer": answer,
        "safety_status": SAFETY_STATUS_ANSWERED,
    }


def _eval_log_node(state: QAState) -> QAState:
    """Log quality signals for observability after answer generation.

    Args:
        state: Current graph state containing final outputs and retrieval metrics.

    Returns:
        QAState: Minimal state write to satisfy LangGraph update requirements.
    """
    logger.info(
        "QA evaluation | request_id=%s | safety_status=%s | top_score=%.4f | match_count=%s | answer_chars=%s",
        state.get("request_id"),
        state.get("safety_status"),
        state.get("top_score", 0.0),
        len(state.get("matches", [])),
        len(state.get("answer", "")),
    )
    # LangGraph requires each node to write at least one state key.
    return {"top_score": state.get("top_score", 0.0)}


def _route_after_safety(state: QAState) -> str:
    """Route to retrieval or terminate based on safety decision."""
    if state.get("safety_status") in {SAFETY_STATUS_BLOCKED, SAFETY_STATUS_OFF_TOPIC}:
        return "end"
    return "retrieve"


def _route_after_confidence(state: QAState) -> str:
    """Route to answer generation or terminate based on confidence gate."""
    if state.get("safety_status") == SAFETY_STATUS_INSUFFICIENT:
        return "end"
    return "answer"


def _build_sources(matches: list[AskMatch], source_json: str) -> list[AskSource]:
    """Convert retrieval matches into API source reference objects.

    Args:
        matches: Retrieved chunks ordered by similarity.
        source_json: JSON artifact path from metadata for source naming.

    Returns:
        list[AskSource]: Source entries including page range, title, and excerpt.
    """
    source_name = Path(source_json).name if source_json else "source.pdf"
    sources: list[AskSource] = []
    for match in matches:
        if match.page_start == match.page_end:
            source_id = f"{source_name}#page={match.page_start}"
        else:
            source_id = f"{source_name}#page={match.page_start}-{match.page_end}"
        sources.append(
            AskSource(
                source_id=source_id,
                title=match.section_heading,
                page=match.page_start,
                page_start=match.page_start,
                page_end=match.page_end,
                excerpt=match.text[:MAX_SOURCE_EXCERPT_CHARS],
            )
        )
    return sources


def _build_graph():
    """Build and compile the LangGraph workflow for QA orchestration."""
    graph = StateGraph(QAState)
    graph.add_node("safety_step", _safety_node)
    graph.add_node("retrieve_step", _retrieve_node)
    graph.add_node("confidence_step", _confidence_node)
    graph.add_node("answer_step", _answer_node)
    graph.add_node("eval_log_step", _eval_log_node)

    graph.set_entry_point("safety_step")
    graph.add_conditional_edges(
        "safety_step",
        _route_after_safety,
        {
            "retrieve": "retrieve_step",
            "end": END,
        },
    )
    graph.add_edge("retrieve_step", "confidence_step")
    graph.add_conditional_edges(
        "confidence_step",
        _route_after_confidence,
        {
            "answer": "answer_step",
            "end": END,
        },
    )
    graph.add_edge("answer_step", "eval_log_step")
    graph.add_edge("eval_log_step", END)

    return graph.compile()


QA_GRAPH = _build_graph()


def run_qa_graph(request: AskRequest, request_id: str) -> AskResponse:
    """Execute the QA graph with local session persistence and checkpoints.

    Args:
        request: Ask request payload from the API.
        request_id: API request id for traceability.

    Returns:
        AskResponse: Final answer payload including sources and safety status.
    """
    initialize_session_store()
    thread_id = get_or_create_thread_id(request.student_id)
    recent_turns = get_recent_turns(request.student_id, SESSION_HISTORY_LIMIT)

    final_state = QA_GRAPH.invoke(
        {
            "request": request,
            "request_id": request_id,
            "thread_id": thread_id,
            "recent_turns": recent_turns,
        },
        config={"configurable": {"thread_id": thread_id}},
    )
    response = AskResponse(
        request_id=request_id,
        answer=final_state.get("answer", ""),
        sources=final_state.get("sources", []),
        safety_status=final_state.get("safety_status", SAFETY_STATUS_INSUFFICIENT),
    )
    save_turn(
        student_id=request.student_id,
        request_id=request_id,
        question=request.question,
        answer=response.answer,
        safety_status=response.safety_status,
        top_score=float(final_state.get("top_score", 0.0)),
    )
    save_checkpoint(
        thread_id=thread_id,
        request_id=request_id,
        state=_serialize_state_for_checkpoint(final_state),
    )
    return response


def _serialize_state_for_checkpoint(state: QAState) -> dict:
    """Convert graph state into a JSON-serializable checkpoint payload.

    Args:
        state: Final graph state containing typed objects.

    Returns:
        dict: Serialized state suitable for SQLite checkpoint storage.
    """
    serialized: dict = {}
    for key, value in state.items():
        if isinstance(value, AskRequest):
            serialized[key] = value.model_dump()
        elif isinstance(value, list) and value and isinstance(value[0], AskMatch):
            serialized[key] = [match.model_dump() for match in value]
        elif isinstance(value, list) and value and isinstance(value[0], AskSource):
            serialized[key] = [source.model_dump() for source in value]
        else:
            serialized[key] = value
    return serialized
