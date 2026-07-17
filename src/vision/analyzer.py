"""Vision understanding - analyze images via Ollama multimodal models."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional


class VisionAnalyzer:
    def __init__(self, model: str = "gemma4:latest") -> None:
        self.model = model

    def _image_to_base64(self, image_path: str | Path) -> str:
        path = Path(image_path)
        return base64.b64encode(path.read_bytes()).decode()

    def analyze(self, image_path: str | Path, prompt: str = "Describe this image in detail.") -> str:
        from src.llm.ollama import OllamaBackend
        import asyncio

        b64 = self._image_to_base64(image_path)
        backend = OllamaBackend()
        messages = [{"role": "user", "content": prompt}]

        collected = ""
        async def _run():
            nonlocal collected
            async for chunk in backend.chat_raw(messages=messages, model=self.model, images=[b64]):
                if "content" in chunk:
                    collected += chunk["content"]
                if chunk.get("done"):
                    break

        asyncio.run(_run())
        return collected

    async def analyze_async(self, image_path: str | Path, prompt: str = "Describe this image in detail.") -> str:
        import base64 as _b64
        from src.llm.ollama import OllamaBackend

        path = Path(image_path)
        b64 = _b64.b64encode(path.read_bytes()).decode()
        backend = OllamaBackend()
        messages = [{"role": "user", "content": prompt}]

        collected = ""
        async for chunk in backend.chat_raw(messages=messages, model=self.model, images=[b64]):
            if "content" in chunk:
                collected += chunk["content"]
            if chunk.get("done"):
                break
        return collected

    def is_multimodal_available(self) -> bool:
        from src.llm.ollama import OllamaBackend
        import asyncio
        backend = OllamaBackend()

        async def _check():
            models = await backend.list_models()
            multimodal_kws = ("gemma", "llava", "bakllava", "moondream", "minicpm")
            return any(any(kw in m.get("name", "").lower() for kw in multimodal_kws) for m in models)

        try:
            return asyncio.run(_check())
        except Exception:
            return False
