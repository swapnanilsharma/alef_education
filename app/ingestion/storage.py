"""Storage utilities for writing extraction and vector store artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.schemas import ExtractedPdf


def sanitize_name(base_name: str) -> str:
    """Return a filesystem-safe artifact name.

    Args:
        base_name: Raw artifact name.

    Returns:
        str: Sanitized artifact name suitable for local file paths.
    """
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in base_name)


def build_output_path(output_dir: Path, base_name: str) -> Path:
    """Build an output JSON path with a sanitized base name and UTC timestamp.

    Args:
        output_dir: Directory where extracted JSON files are written.
        base_name: Preferred source name used to generate the filename prefix.

    Returns:
        Path: Full destination path for the JSON output.
    """
    safe_name = sanitize_name(base_name)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return output_dir / f"{safe_name}_{timestamp}.json"


def build_artifact_stem(base_dir: Path, base_name: str) -> Path:
    """Build a timestamped base path for related artifacts.

    Args:
        base_dir: Directory where artifacts are stored.
        base_name: Preferred source name for the artifact prefix.

    Returns:
        Path: Base path without file extension.
    """
    safe_name = sanitize_name(base_name)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return base_dir / f"{safe_name}_{timestamp}"


def save_to_json(data: ExtractedPdf, output_path: Path) -> None:
    """Persist extracted PDF content to a JSON file.

    Args:
        data: Parsed PDF content represented as an ExtractedPdf model.
        output_path: Destination path for the JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data.model_dump(), file, ensure_ascii=False, indent=2)


def load_json_file(input_path: Path) -> dict[str, Any]:
    """Load a JSON document from disk.

    Args:
        input_path: Path to the JSON file.

    Returns:
        dict[str, Any]: Parsed JSON payload.
    """
    with input_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json_payload(data: dict[str, Any], output_path: Path) -> None:
    """Persist an arbitrary JSON payload to disk.

    Args:
        data: JSON-serializable payload.
        output_path: Destination path for the JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
