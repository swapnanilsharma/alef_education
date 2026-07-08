"""Utilities to clean PDF text while preserving math notation."""

from __future__ import annotations

import re

_MATH_SYMBOLS = set("=+-*/^%()[]{}<>|\\:~")
_UNICODE_MATH_SYMBOLS = {"≤", "≥", "≠", "≈", "±", "√", "∞", "∑", "∫", "π"}


def _looks_like_math(line: str) -> bool:
    """Heuristically detect equation-like content so it is not over-normalized."""
    if not line:
        return False

    if any(symbol in line for symbol in _UNICODE_MATH_SYMBOLS):
        return True

    symbol_hits = sum(1 for char in line if char in _MATH_SYMBOLS)
    has_digits = any(char.isdigit() for char in line)
    has_alpha = any(char.isalpha() for char in line)

    if symbol_hits >= 2 and (has_digits or has_alpha):
        return True

    if re.search(r"\b[a-zA-Z]\s*[=<>]\s*[-+]?\d", line):
        return True

    return bool(re.search(r"\b\d+\s*[+\-*/=]\s*\d+\b", line))


def clean_page_text(raw_text: str) -> str:
    """Remove common PDF extraction noise while keeping formulas intact.

    The cleaner intentionally avoids aggressive transformations on equation-like
    lines so mathematical expressions remain suitable for embedding later.

    Args:
        raw_text: Raw page text from PyMuPDF.

    Returns:
        str: Normalized, lower-noise page text.
    """
    normalized = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u00a0", " ").replace("\u200b", "")

    lines = [line.strip() for line in normalized.split("\n")]

    filtered_lines: list[str] = []
    for line in lines:
        if not line:
            filtered_lines.append("")
            continue

        if re.fullmatch(r"\d+", line):
            continue

        if len(line) <= 2 and not _looks_like_math(line):
            continue

        compact = re.sub(r"\s+", " ", line)
        filtered_lines.append(compact)

    cleaned_lines: list[str] = []
    index = 0
    while index < len(filtered_lines):
        line = filtered_lines[index]
        if (
            line
            and line.endswith("-")
            and len(line) > 1
            and line[-2].isalpha()
            and index + 1 < len(filtered_lines)
            and filtered_lines[index + 1]
            and filtered_lines[index + 1][0].islower()
            and not _looks_like_math(line)
            and not _looks_like_math(filtered_lines[index + 1])
        ):
            cleaned_lines.append(line[:-1] + filtered_lines[index + 1])
            index += 2
            continue

        cleaned_lines.append(line)
        index += 1

    output_lines: list[str] = []
    blank_seen = False
    for line in cleaned_lines:
        if not line:
            if not blank_seen:
                output_lines.append("")
            blank_seen = True
            continue

        output_lines.append(line)
        blank_seen = False

    return "\n".join(output_lines).strip()
