from __future__ import annotations

import asyncio
import json
import threading
from typing import AsyncGenerator, Optional
from urllib.request import Request, urlopen
import socket
from urllib.error import URLError

from src.config.settings import get_settings

_SENTINEL = object()


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
        except socket.timeout:
            raise RuntimeError(f"Ollama timed out after {timeout}s — model may be too large for this system") from None

    async def _stream_async(self, url: str, data: bytes, timeout: Optional[int] = None) -> AsyncGenerator[dict, None]:
        timeout = timeout or self.settings.ollama_request_timeout
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _read() -> None:
            req = Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
            try:
                with urlopen(req, timeout=timeout) as resp:
                    for line in resp:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        loop.call_soon_threadsafe(queue.put_nowait, item)
            except URLError as e:
                loop.call_soon_threadsafe(queue.put_nowait, e)
            except OSError as e:
                loop.call_soon_threadsafe(queue.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

        thread = threading.Thread(target=_read, daemon=True)
        thread.start()

        while True:
            item = await queue.get()
            if item is _SENTINEL:
                break
            if isinstance(item, Exception):
                raise RuntimeError(f"Ollama stream request failed: {item}") from item
            yield item

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
        model = model or self.settings.default_model
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": True,
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

        async for item in self._stream_async(self._build_url("/api/generate"), data, None):
            if "response" in item:
                yield item["response"]
            if item.get("done"):
                break

    async def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        images: Optional[list[str]] = None,
        stream: bool = True,
    ) -> AsyncGenerator[str, None]:
        model = model or self.settings.default_model
        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature if temperature is not None else self.settings.temperature,
                "num_predict": max_tokens if max_tokens is not None else self.settings.max_tokens,
            },
        }

        if images:
            last_msg = payload["messages"][-1]
            if last_msg.get("role") == "user":
                last_msg["images"] = images

        data = json.dumps(payload).encode()

        async for item in self._stream_async(self._build_url("/api/chat"), data, None):
            if "message" in item and "content" in item["message"]:
                yield item["message"]["content"]
            if item.get("done"):
                break

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
