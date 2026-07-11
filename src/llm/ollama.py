from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

from src.config.settings import get_settings


class OllamaBackend:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_url = self.settings.ollama_host.rstrip("/")

    def _build_url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _request_sync(self, url: str, data: Optional[bytes] = None, timeout: Optional[int] = None) -> dict:
        timeout = timeout or self.settings.ollama_request_timeout
        req = Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
        try:
            with urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except URLError as e:
            raise RuntimeError(f"Ollama request failed: {e.reason}") from e

    def _stream_sync(self, url: str, data: bytes, timeout: Optional[int] = None) -> list[dict]:
        timeout = timeout or self.settings.ollama_request_timeout
        req = Request(url, data=data, headers={"Content-Type": "application/json"})
        results: list[dict] = []
        try:
            with urlopen(req, timeout=timeout) as resp:
                for line in resp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except URLError as e:
            raise RuntimeError(f"Ollama stream request failed: {e.reason}") from e
        return results

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stream: bool = False,
    ) -> AsyncGenerator[str, None]:
        model = model or self.settings.default_model
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature if temperature is not None else self.settings.temperature,
                "num_predict": max_tokens if max_tokens is not None else self.settings.max_tokens,
            },
        }
        if system:
            payload["system"] = system
        if top_k is not None:
            payload["options"]["top_k"] = top_k
        if top_p is not None:
            payload["options"]["top_p"] = top_p

        data = json.dumps(payload).encode()

        loop = asyncio.get_event_loop()
        if stream:
            results = await loop.run_in_executor(None, self._stream_sync, self._build_url("/api/generate"), data, None)
            for item in results:
                if "response" in item:
                    yield item["response"]
                if item.get("done"):
                    break
        else:
            result = await loop.run_in_executor(None, self._request_sync, self._build_url("/api/generate"), data)
            yield result.get("response", "")

    async def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> AsyncGenerator[str, None]:
        model = model or self.settings.default_model
        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature if temperature is not None else self.settings.temperature,
                "num_predict": max_tokens if max_tokens is not None else self.settings.max_tokens,
            },
        }

        data = json.dumps(payload).encode()
        loop = asyncio.get_event_loop()

        if stream:
            results = await loop.run_in_executor(None, self._stream_sync, self._build_url("/api/chat"), data)
            for item in results:
                if "message" in item and "content" in item["message"]:
                    yield item["message"]["content"]
                if item.get("done"):
                    break
        else:
            result = await loop.run_in_executor(None, self._request_sync, self._build_url("/api/chat"), data)
            if "message" in result:
                yield result["message"]["content"]

    async def list_models(self) -> list[dict]:
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._request_sync, self._build_url("/api/tags"), None, 10)
            return result.get("models", [])
        except Exception:
            return []

    async def is_available(self) -> bool:
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._request_sync, self._build_url("/api/tags"), None, 5)
            return True
        except Exception:
            return False
