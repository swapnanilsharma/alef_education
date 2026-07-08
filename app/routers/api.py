"""Router aggregator – combines all sub-routers into a single APIRouter."""

from __future__ import annotations

from fastapi import APIRouter

from app.routers.health import router as health_router
from app.routers.ingest import router as ingest_router
from app.routers.qa import router as qa_router

router = APIRouter()
router.include_router(health_router)
router.include_router(ingest_router)
router.include_router(qa_router)
