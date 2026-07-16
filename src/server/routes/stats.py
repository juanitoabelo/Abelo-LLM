from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from src.config.settings import get_settings
from src.monitor.stats import UsageTracker

router = APIRouter(prefix="/api/stats", tags=["stats"])


def _get_tracker() -> UsageTracker:
    settings = get_settings()
    base_path = Path(settings.data_dir).parent
    return UsageTracker(str(base_path / "usage.db"))


@router.get("")
async def get_stats(hours: int = 24):
    tracker = _get_tracker()
    return tracker.get_stats(hours=hours)


@router.get("/requests")
async def get_recent_requests(limit: int = 20):
    tracker = _get_tracker()
    return {"requests": tracker.get_recent_requests(limit=limit)}
