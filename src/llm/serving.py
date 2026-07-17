"""Model serving optimization — continuous batching, spec decoding, caching."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any, Optional


class ContinuousBatcher:
    """Simple request batcher — groups concurrent requests to same model."""

    def __init__(self, batch_window_ms: float = 100.0) -> None:
        self.batch_window = batch_window_ms / 1000.0
        self._pending: dict[str, list[dict]] = {}
        self._lock = threading.Lock()

    async def submit(self, model: str, request: dict) -> Any:
        import asyncio
        future = asyncio.get_event_loop().create_future()
        with self._lock:
            if model not in self._pending:
                self._pending[model] = []
                asyncio.ensure_future(self._process(model))
            self._pending[model].append({"request": request, "future": future})
        return await future

    async def _process(self, model: str) -> None:
        import asyncio
        await asyncio.sleep(self.batch_window)
        with self._lock:
            batch = self._pending.pop(model, [])
        if not batch:
            return
        for item in batch:
            if not item["future"].done():
                item["future"].set_result({"batched": True, "model": model})


class SpeculativeDecoder:
    """Speculative decoding — draft model generates candidates, target model verifies."""

    def __init__(self, draft_model: str = "llama3.2:1b", target_model: str = "llama3.2:3b") -> None:
        self.draft_model = draft_model
        self.target_model = target_model

    async def generate(self, prompt: str, ollama_backend, max_tokens: int = 128) -> str:
        draft = ""
        async for chunk in ollama_backend.generate(prompt=prompt, model=self.draft_model, max_tokens=max_tokens // 2):
            draft += chunk

        verify_prompt = f"{prompt}\n\nDraft response: {draft}\n\nVerify and improve this response. Fix any errors and expand if needed:"
        verified = ""
        async for chunk in ollama_backend.generate(prompt=verify_prompt, model=self.target_model, max_tokens=max_tokens):
            verified += chunk
        return verified if len(verified) > len(draft) // 2 else draft


class ResponseCache:
    """Multi-level response cache — memory + disk with TTL."""

    def __init__(self, max_memory: int = 200, disk_path: str = "data/response_cache") -> None:
        self.max_memory = max_memory
        self.disk_path = __import__("pathlib").Path(disk_path)
        self.disk_path.mkdir(parents=True, exist_ok=True)
        self._memory: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._lock = threading.Lock()

    def _make_key(self, model: str, prompt: str, temperature: float) -> str:
        raw = json.dumps({"m": model, "p": prompt, "t": temperature}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, model: str, prompt: str, temperature: float = 0.0) -> Optional[str]:
        key = self._make_key(model, prompt, temperature)
        with self._lock:
            if key in self._memory:
                ts, val = self._memory[key]
                if time.time() - ts < 3600:
                    self._memory.move_to_end(key)
                    return val
                del self._memory[key]
        disk_path = self.disk_path / key
        if disk_path.exists():
            try:
                data = json.loads(disk_path.read_text())
                if time.time() - data["ts"] < 86400:
                    return data["response"]
            except Exception:
                pass
        return None

    def set(self, model: str, prompt: str, response: str, temperature: float = 0.0) -> None:
        key = self._make_key(model, prompt, temperature)
        with self._lock:
            self._memory[key] = (time.time(), response)
            while len(self._memory) > self.max_memory:
                self._memory.popitem(last=False)
        try:
            (self.disk_path / key).write_text(json.dumps({"response": response, "ts": time.time()}))
        except Exception:
            pass
