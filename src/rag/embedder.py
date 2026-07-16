from __future__ import annotations

import asyncio
import json
from typing import Optional
from urllib.request import Request, urlopen

from src.config.settings import get_settings


class OllamaEmbedder:
    def __init__(self, model: Optional[str] = None) -> None:
        self.settings = get_settings()
        self.base_url = self.settings.ollama_host.rstrip("/")
        self.model = model or "nomic-embed-text"
        self._dimension: Optional[int] = None

    def _build_url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = self._probe_dimension()
        return self._dimension

    def _probe_dimension(self) -> int:
        result = self._embed_sync(["probe"])
        return len(result[0]) if result else 768

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": self.model, "input": texts}).encode()
        req = Request(
            self._build_url("/api/embed"),
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
                return data.get("embeddings", [])
        except Exception:
            return []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._embed_sync, texts)

    async def embed_query(self, text: str) -> list[float]:
        results = await self.embed([text])
        return results[0] if results else []

    async def is_available(self) -> bool:
        try:
            loop = asyncio.get_event_loop()
            payload = json.dumps({"model": self.model, "input": ["ping"]}).encode()
            req = Request(
                self._build_url("/api/embed"),
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            await loop.run_in_executor(
                None,
                lambda: urlopen(req, timeout=10),
            )
            return True
        except Exception:
            return False
