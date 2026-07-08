"""FastAPI application entrypoint."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from app.core.config import OUTPUTS_DIR, VECTOR_STORE_DIR
from app.core.logging_config import configure_logging
from app.routers.api import router as api_router

configure_logging()

logger = logging.getLogger(__name__)

_REQUIRED_DIRS = [OUTPUTS_DIR, VECTOR_STORE_DIR]

for _dir in _REQUIRED_DIRS:
    _dir.mkdir(parents=True, exist_ok=True)
    logger.info("Directory ensured | path=%s", _dir)

app = FastAPI(title="PDF Ingestion Service", version="1.0.0")
app.include_router(api_router)

logger.info("FastAPI application initialized")
