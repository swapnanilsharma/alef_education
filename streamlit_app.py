"""Streamlit UI for the AI learning assistant backend."""

from __future__ import annotations

from typing import Any

import requests
import streamlit as st

DEFAULT_API_URL = "http://127.0.0.1:8000"


@st.cache_data(show_spinner=False)
def call_health(base_url: str) -> dict[str, Any]:
    """Call the backend health endpoint and return the JSON payload.

    Args:
        base_url: Base URL where the FastAPI backend is running.

    Returns:
        dict[str, Any]: Parsed JSON response from the health endpoint.
    """
    response = requests.get(f"{base_url.rstrip('/')}/health", timeout=10)
    response.raise_for_status()
    return response.json()


def call_ingest(base_url: str, pdf_name: str, pdf_bytes: bytes) -> dict[str, Any]:
    """Upload a PDF to the backend ingest endpoint.

    Args:
        base_url: Base URL where the FastAPI backend is running.
        pdf_name: Original filename for the uploaded PDF.
        pdf_bytes: Raw PDF bytes to send as multipart form data.

    Returns:
        dict[str, Any]: Parsed JSON response from the ingest endpoint.
    """
    files = {"file": (pdf_name, pdf_bytes, "application/pdf")}
    response = requests.post(f"{base_url.rstrip('/')}/ingest", files=files, timeout=180)
    response.raise_for_status()
    return response.json()


def call_ask(base_url: str, student_id: str, question: str, grade_level: str) -> dict[str, Any]:
    """Send a student question to the backend ask endpoint.

    Args:
        base_url: Base URL where the FastAPI backend is running.
        student_id: Student identifier used for local session tracking.
        question: Student question text.
        grade_level: Grade level used to adapt the answer.

    Returns:
        dict[str, Any]: Parsed JSON response from the ask endpoint.
    """
    payload = {
        "student_id": student_id,
        "question": question,
        "grade_level": grade_level,
    }
    response = requests.post(f"{base_url.rstrip('/')}/ask", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def show_sources(sources: list[dict[str, Any]]) -> None:
    """Render answer source references in expandable Streamlit blocks.

    Args:
        sources: Source payloads returned by the ask endpoint.
    """
    if not sources:
        st.info("No sources returned.")
        return

    st.subheader("Sources")
    for idx, source in enumerate(sources, start=1):
        title = source.get("title", "Unknown")
        page = _format_source_page(source)
        source_id = source.get("source_id", "")
        excerpt = source.get("excerpt", "")
        with st.expander(f"{idx}. {title} (Page {page})", expanded=False):
            if source_id:
                st.caption(source_id)
            st.write(excerpt)


def _format_source_page(source: dict[str, Any]) -> str:
    """Format either a single page or a page range for UI display."""
    page = source.get("page")
    if page is not None:
        return str(page)

    page_start = source.get("page_start")
    page_end = source.get("page_end")
    if page_start is None or page_end is None:
        return "?"
    if page_start == page_end:
        return str(page_start)
    return f"{page_start}-{page_end}"


def main() -> None:
    """Render the Streamlit application for health, ingest, and ask workflows."""
    st.set_page_config(page_title="AI Learning Assistant", page_icon="📘", layout="wide")
    st.title("AI Learning Assistant")
    st.caption("UI for /health, /ingest, and /ask APIs")

    if "ask_history" not in st.session_state:
        st.session_state.ask_history = []

    with st.sidebar:
        st.header("Connection")
        base_url = st.text_input("Backend URL", value=DEFAULT_API_URL)
        st.markdown("---")
        st.header("Ask Defaults")
        default_student_id = st.text_input("Student ID", value="student-123")
        default_grade = st.selectbox(
            "Grade Level",
            options=["grade_6", "grade_7", "grade_8", "grade_9", "grade_10", "grade_11", "grade_12"],
            index=2,
        )

    tab_health, tab_ingest, tab_ask = st.tabs(["Health", "Ingest PDF", "Ask"])

    with tab_health:
        st.subheader("Service Health")
        if st.button("Check /health", type="primary"):
            try:
                payload = call_health(base_url)
                st.success("Backend is healthy")
                st.json(payload)
            except requests.RequestException as exc:
                st.error(f"Health check failed: {exc}")

    with tab_ingest:
        st.subheader("Ingest PDF")
        uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
        ingest_clicked = st.button("Send to /ingest", type="primary", disabled=uploaded is None)

        if ingest_clicked and uploaded is not None:
            with st.spinner("Ingesting PDF and building embeddings..."):
                try:
                    payload = call_ingest(base_url, uploaded.name, uploaded.getvalue())
                    st.success("Ingest completed")
                    st.json(payload)
                except requests.HTTPError as exc:
                    detail = exc.response.text if exc.response is not None else str(exc)
                    st.error(f"Ingest failed: {detail}")
                except requests.RequestException as exc:
                    st.error(f"Ingest failed: {exc}")

    with tab_ask:
        st.subheader("Ask a Question")
        with st.form("ask_form"):
            student_id = st.text_input("Student ID", value=default_student_id)
            grade_level = st.selectbox(
                "Grade Level",
                options=["grade_6", "grade_7", "grade_8", "grade_9", "grade_10", "grade_11", "grade_12"],
                index=["grade_6", "grade_7", "grade_8", "grade_9", "grade_10", "grade_11", "grade_12"].index(default_grade),
            )
            question = st.text_area("Question", placeholder="How do I solve 4x - 7 = 21?", height=120)
            submitted = st.form_submit_button("Send to /ask", type="primary")

        if submitted:
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Generating answer..."):
                    try:
                        payload = call_ask(base_url, student_id.strip(), question.strip(), grade_level)
                        st.session_state.ask_history.insert(0, payload)
                        st.success("Answer received")

                        st.subheader("Answer")
                        st.write(payload.get("answer", ""))

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Safety Status", payload.get("safety_status", "unknown"))
                        with col2:
                            st.caption(f"Request ID: {payload.get('request_id', '')}")

                        show_sources(payload.get("sources", []))
                    except requests.HTTPError as exc:
                        detail = exc.response.text if exc.response is not None else str(exc)
                        st.error(f"Ask failed: {detail}")
                    except requests.RequestException as exc:
                        st.error(f"Ask failed: {exc}")

        st.markdown("---")
        st.subheader("Recent Ask Responses")
        if not st.session_state.ask_history:
            st.caption("No /ask responses in this UI session yet.")
        else:
            for item in st.session_state.ask_history[:10]:
                with st.expander(
                    f"{item.get('request_id', 'request')} | {item.get('safety_status', 'unknown')}",
                    expanded=False,
                ):
                    st.write(item.get("answer", ""))
                    show_sources(item.get("sources", []))


if __name__ == "__main__":
    main()
