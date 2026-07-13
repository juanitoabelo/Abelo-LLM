from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Optional

from src.config.settings import get_settings
from src.llm.ollama import OllamaBackend
from src.llm.local_model import LocalModelBackend


class LLMRouter:
    WINNER_KEYWORDS = ("winner", "custom", "tinytransformer")

    def __init__(self) -> None:
        self.settings = get_settings()
        self.ollama = OllamaBackend()
        self.local = LocalModelBackend() if self.settings.enable_custom_model else None
        self._backend_available: Optional[bool] = None

    def _resolve_backend(self, model: Optional[str] = None) -> OllamaBackend | LocalModelBackend:
        if model and any(kw in model.lower() for kw in self.WINNER_KEYWORDS):
            return self.local or self.ollama
        if model and self.local and self.settings.enable_custom_model:
            return self.ollama
        if self.local and self.settings.enable_custom_model and not model:
            return self.local
        return self.ollama

    async def check_backends(self) -> dict[str, bool]:
        ollama_ok = await self.ollama.is_available()
        local_ok = await self.local.is_available() if self.local else False
        self._backend_available = ollama_ok or local_ok
        return {"ollama": ollama_ok, "local": local_ok}

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stream: bool = True,
    ) -> AsyncGenerator[str, None]:
        backend = self._resolve_backend(model)
        if hasattr(backend, 'generate') and not hasattr(backend, 'chat'):
            async for chunk in backend.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_k=top_k,
                top_p=top_p,
            ):
                yield chunk
        else:
            async for chunk in backend.generate(
                prompt=prompt,
                model=model,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                top_k=top_k,
                top_p=top_p,
                stream=stream,
            ):
                yield chunk

    async def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        images: Optional[list[str]] = None,
        stream: bool = True,
    ) -> AsyncGenerator[str, None]:
        backend = self._resolve_backend(model)
        if backend is self.ollama:
            async for chunk in backend.chat(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                images=images,
                stream=stream,
            ):
                yield chunk
        else:
            user_msg = messages[-1]["content"] if messages else ""
            async for chunk in backend.generate(
                prompt=user_msg,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                yield chunk

    async def list_models(self) -> list[dict]:
        ollama_models = await self.ollama.list_models()
        models = []
        for m in ollama_models:
            name = m.get("name", "")
            models.append({
                "name": name,
                "size": m.get("size", 0),
                "provider": "ollama",
            })
        if self.local and self.settings.enable_custom_model:
            models.insert(0, {"name": "winner-model (custom)", "size": 0, "provider": "local"})
        return models

    def format_chat_messages(self, system_prompt: Optional[str], history: list[dict], user_message: str) -> list[dict]:
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for msg in history:
            messages.append(msg)
        messages.append({"role": "user", "content": user_message})
        return messages
