from __future__ import annotations

import time

from fastapi import APIRouter

from src.llm.router import LLMRouter
from src.llm.registry import ModelRegistry

router = APIRouter(prefix="/api/models", tags=["models"])


def _get_registry() -> ModelRegistry:
    return ModelRegistry()


@router.get("")
async def list_models():
    llm = LLMRouter()
    backends = await llm.check_backends()
    registry = _get_registry()

    models = []
    if backends.get("ollama"):
        ollama_models = await llm.list_models()
        registry.sync_from_ollama(ollama_models)
        for m in ollama_models:
            models.append(m)

    if backends.get("local"):
        if llm.local and llm.settings.enable_custom_model:
            models.append({"name": "winner-model (custom)", "size": 0, "provider": "local"})

    return {"backends": backends, "models": models}


@router.get("/health")
async def health_check():
    llm = LLMRouter()
    backends = await llm.check_backends()
    registry = _get_registry()

    health = {
        "status": "ok" if any(backends.values()) else "degraded",
        "backends": backends,
    }

    if backends.get("ollama"):
        try:
            t0 = time.time()
            await llm.ollama.is_available()
            latency = (time.time() - t0) * 1000
            health["ollama_latency_ms"] = round(latency, 1)
        except Exception:
            pass

    registry_summary = registry.get_health_summary()
    health["registry"] = registry_summary

    return health


@router.get("/registry")
async def model_registry():
    registry = _get_registry()
    models = registry.list_models()
    return {
        "models": [m.to_dict() for m in models],
        "summary": registry.get_health_summary(),
    }


@router.get("/{name}")
async def get_model_info(name: str):
    registry = _get_registry()
    model = registry.get_model(name)
    if not model:
        return {"error": f"Model '{name}' not found"}
    return {"model": model.to_dict()}
