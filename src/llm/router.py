from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Optional

from src.config.settings import get_settings
from src.llm.ollama import OllamaBackend


class LLMRouter:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.ollama = OllamaBackend()
        self._backend_available: Optional[bool] = None

    async def check_backends(self) -> dict[str, bool]:
        ollama_ok = await self.ollama.is_available()
        self._backend_available = ollama_ok
        return {"ollama": ollama_ok}

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
        async for chunk in self.ollama.generate(
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
        stream: bool = True,
    ) -> AsyncGenerator[str, None]:
        async for chunk in self.ollama.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        ):
            yield chunk

    async def list_models(self) -> list[dict]:
        return await self.ollama.list_models()

    def format_chat_messages(self, system_prompt: Optional[str], history: list[dict], user_message: str) -> list[dict]:
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for msg in history:
            messages.append(msg)
        messages.append({"role": "user", "content": user_message})
        return messages
