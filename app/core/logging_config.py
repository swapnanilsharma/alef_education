"""Central logging configuration for the service."""

from __future__ import annotations

import logging
import os


def configure_logging() -> None:
    """Configure application-wide logging once at startup."""
    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(log_level)
        return

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
