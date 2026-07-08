"""Health check endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health")
async def health() -> dict[str, str]:
    """Return service health status for uptime probes.

    Returns:
        dict[str, str]: A minimal status payload indicating the API is healthy.
    """
    logger.debug("Health check requested")
    return {"status": "ok"}
