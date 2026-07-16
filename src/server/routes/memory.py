from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from src.config.settings import get_settings
from src.memory.store import MemoryStore

router = APIRouter(prefix="/api/memory", tags=["memory"])


def _get_store() -> MemoryStore:
    settings = get_settings()
    base_path = Path(settings.data_dir).parent
    return MemoryStore(str(base_path / "memory.db"))


@router.get("")
async def list_memories(namespace: str = "default"):
    store = _get_store()
    memories = store.get_all_memories(namespace)
    return {"memories": memories}


class RememberRequest(BaseModel):
    key: str
    value: str
    namespace: str = "default"
    importance: float = 1.0


@router.post("/remember")
async def remember(request: RememberRequest):
    store = _get_store()
    store.remember(request.key, request.value, request.namespace, request.importance)
    return {"status": "ok"}


@router.get("/recall/{key}")
async def recall(key: str, namespace: str = "default"):
    store = _get_store()
    value = store.recall(key, namespace)
    return {"key": key, "value": value}


@router.delete("/forget/{key}")
async def forget(key: str, namespace: str = "default"):
    store = _get_store()
    store.forget(key, namespace)
    return {"status": "forgotten", "key": key}


@router.get("/sessions")
async def list_sessions():
    store = _get_store()
    sessions = store.get_recent_sessions(limit=10)
    return {"sessions": sessions}
