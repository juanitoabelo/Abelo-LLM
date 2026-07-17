"""Simple LRU prompt cache to avoid re-processing identical requests."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any, Optional


class PromptCache:
    """Thread-safe LRU cache for LLM prompt responses."""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 300) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._lock = threading.Lock()

    def _make_key(self, prompt: str, model: Optional[str] = None, temperature: Optional[float] = None) -> str:
        data = {"p": prompt, "m": model, "t": temperature}
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, prompt: str, model: Optional[str] = None, temperature: Optional[float] = None) -> Optional[str]:
        key = self._make_key(prompt, model, temperature)
        with self._lock:
            if key not in self._cache:
                return None
            timestamp, response = self._cache[key]
            if time.time() - timestamp > self.ttl_seconds:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return response

    def set(self, prompt: str, response: str, model: Optional[str] = None, temperature: Optional[float] = None) -> None:
        key = self._make_key(prompt, model, temperature)
        with self._lock:
            self._cache[key] = (time.time(), response)
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    def invalidate(self, prompt: str, model: Optional[str] = None, temperature: Optional[float] = None) -> None:
        key = self._make_key(prompt, model, temperature)
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)


_cache: Optional[PromptCache] = None


def get_cache() -> PromptCache:
    global _cache
    if _cache is None:
        _cache = PromptCache()
    return _cache
