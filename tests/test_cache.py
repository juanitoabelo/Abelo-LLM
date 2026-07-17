from __future__ import annotations

from src.cache.prompt_cache import PromptCache


class TestPromptCache:
    def test_get_miss(self) -> None:
        cache = PromptCache(max_size=10, ttl_seconds=60)
        result = cache.get("hello")
        assert result is None

    def test_set_and_get(self) -> None:
        cache = PromptCache(max_size=10, ttl_seconds=60)
        cache.set("hello", "world")
        result = cache.get("hello")
        assert result == "world"

    def test_cache_key_variation(self) -> None:
        cache = PromptCache(max_size=10, ttl_seconds=60)
        cache.set("hello", "world", model="model_a")
        assert cache.get("hello", model="model_b") is None
        assert cache.get("hello", model="model_a") == "world"

    def test_invalidate(self) -> None:
        cache = PromptCache(max_size=10, ttl_seconds=60)
        cache.set("key", "value")
        cache.invalidate("key")
        assert cache.get("key") is None

    def test_max_size(self) -> None:
        cache = PromptCache(max_size=3, ttl_seconds=60)
        cache.set("a", "1")
        cache.set("b", "2")
        cache.set("c", "3")
        cache.set("d", "4")
        assert cache.size <= 3

    def test_clear(self) -> None:
        cache = PromptCache(max_size=10, ttl_seconds=60)
        cache.set("a", "1")
        cache.set("b", "2")
        cache.clear()
        assert cache.size == 0

    def test_ttl_expiry(self) -> None:
        cache = PromptCache(max_size=10, ttl_seconds=0)
        cache.set("quick", "expire")
        import time
        time.sleep(0.01)
        assert cache.get("quick") is None

    def test_different_temperature(self) -> None:
        cache = PromptCache(max_size=10, ttl_seconds=60)
        cache.set("hello", "warm", temperature=0.7)
        assert cache.get("hello", temperature=0.8) is None
        assert cache.get("hello", temperature=0.7) == "warm"
