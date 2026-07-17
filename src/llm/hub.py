"""Model hub integration — pull/push from Hugging Face, community discovery."""

from __future__ import annotations

import json
import time
from typing import Optional


class ModelHub:
    def __init__(self) -> None:
        self._cache: dict[str, list[dict]] = {}
        self._cache_time: float = 0

    async def search_huggingface(self, query: str, limit: int = 20) -> list[dict]:
        import httpx

        url = f"https://huggingface.co/api/models?search={query}&sort=downloads&direction=-1&limit={limit}"
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(url)
                if r.status_code == 200:
                    data = r.json()
                    results = []
                    for m in data:
                        results.append({
                            "id": m.get("modelId", ""),
                            "pipeline_tag": m.get("pipeline_tag", ""),
                            "downloads": m.get("downloads", 0),
                            "likes": m.get("likes", 0),
                            "library": (m.get("library_name") or [None])[0] if isinstance(m.get("library_name"), list) else m.get("library_name"),
                            "tags": m.get("tags", []),
                        })
                    return results
        except Exception:
            pass
        return []

    async def search_ollama_library(self, query: str = "") -> list[dict]:
        import httpx

        url = "https://ollama.ai/api/library" if not query else f"https://ollama.ai/api/library?q={query}"
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(url)
                if r.status_code == 200:
                    data = r.json()
                    return data if isinstance(data, list) else data.get("models", [])
        except Exception:
            pass
        return []

    async def get_model_details(self, model_id: str) -> Optional[dict]:
        import httpx

        if ":" in model_id or not model_id.startswith("http"):
            url = f"https://huggingface.co/api/models/{model_id.replace('/', '/').split(':')[0]}"
        else:
            url = model_id

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(url)
                if r.status_code == 200:
                    return r.json()
        except Exception:
            pass
        return None

    async def suggest_models(self, task: str = "chat") -> list[dict]:
        recommendations = {
            "chat": [
                {"id": "llama3.2:1b", "size": "1.3GB", "description": "Fast, lightweight chat"},
                {"id": "llama3.2:3b", "size": "2.0GB", "description": "Balanced chat model"},
                {"id": "qwen3.5:latest", "size": "3.2GB", "description": "Strong general purpose"},
                {"id": "gemma4:latest", "size": "2.5GB", "description": "Google multimodal"},
            ],
            "code": [
                {"id": "deepseek-coder:latest", "size": "3.5GB", "description": "Code generation"},
                {"id": "codellama:latest", "size": "3.8GB", "description": "General code assistant"},
            ],
            "embedding": [
                {"id": "nomic-embed-text", "size": "0.3GB", "description": "Text embeddings"},
                {"id": "snowflake-arctic-embed", "size": "0.4GB", "description": "High quality embeddings"},
            ],
            "vision": [
                {"id": "llava:latest", "size": "4.5GB", "description": "Vision-language model"},
                {"id": "gemma4:latest", "size": "2.5GB", "description": "Multimodal with vision"},
            ],
        }
        return recommendations.get(task, recommendations["chat"])
