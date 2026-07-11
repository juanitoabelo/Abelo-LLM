from __future__ import annotations

from fastapi import APIRouter

from src.llm.router import LLMRouter

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("")
async def list_models():
    llm = LLMRouter()
    backends = await llm.check_backends()
    models = []
    if backends.get("ollama"):
        models = await llm.list_models()
    return {"backends": backends, "models": models}


@router.get("/health")
async def health_check():
    llm = LLMRouter()
    backends = await llm.check_backends()
    return {"status": "ok", "backends": backends}
