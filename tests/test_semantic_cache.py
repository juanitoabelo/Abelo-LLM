"""Tests for semantic cache."""

import numpy as np
import pytest
from src.llm.semantic_cache import SemanticCache


@pytest.fixture
def cache(tmp_path):
    return SemanticCache(persist_path=str(tmp_path / "cache.json"), max_entries=10)


def test_cache_set_get(cache):
    emb = [0.1, 0.2, 0.3, 0.4]
    cache.set("hello", emb, "hi there!")
    result = cache.get(emb)
    assert result == "hi there!"


def test_cache_similar(cache):
    emb1 = [1.0, 0.0]
    emb2 = [0.99, 0.02]
    cache.set("a", emb1, "response_a")
    result = cache.get(emb2)
    assert result == "response_a"


def test_cache_miss(cache):
    emb1 = [1.0, 0.0]
    emb2 = [0.0, 1.0]
    cache.set("a", emb1, "resp")
    result = cache.get(emb2)
    assert result is None


def test_cache_model_filter(cache):
    cache.set("hello", [0.1, 0.2], "llama response", model="llama")
    cache.set("hello", [0.1, 0.2], "qwen response", model="qwen")
    result = cache.get([0.1, 0.2], model="llama")
    assert result == "llama response"
    result2 = cache.get([0.1, 0.2], model="qwen")
    assert result2 == "qwen response"


def test_cache_stats(cache):
    cache.set("q1", [0.1], "a1")
    cache.set("q2", [0.2], "a2")
    s = cache.stats()
    assert s["entries"] == 2


def test_cache_clear(cache):
    cache.set("q1", [0.1], "a1")
    cache.clear()
    assert cache.stats()["entries"] == 0


def test_normalize(cache):
    vec = [3.0, 4.0]
    normalized = cache._normalize(vec)
    assert abs(float(np.linalg.norm(normalized)) - 1.0) < 0.001


def test_cosine_similarity(cache):
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert abs(cache._cosine_similarity(a, b)) < 0.001
    assert abs(cache._cosine_similarity(a, a) - 1.0) < 0.001
