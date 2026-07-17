"""Tests for model serving optimizations."""

import time
import pytest
from src.llm.serving import ResponseCache, SpeculativeDecoder, ContinuousBatcher


@pytest.fixture
def cache(tmp_path):
    return ResponseCache(max_memory=10, disk_path=str(tmp_path / "cache"))


def test_cache_set_get(cache):
    cache.set("model1", "Hello", "Hi there!", temperature=0.0)
    result = cache.get("model1", "Hello", temperature=0.0)
    assert result == "Hi there!"


def test_cache_miss(cache):
    result = cache.get("nonexistent", "nope", temperature=0.5)
    assert result is None


def test_cache_different_temp(cache):
    cache.set("m", "Hello", "Response for temp 0.0", temperature=0.0)
    result = cache.get("m", "Hello", temperature=0.7)
    assert result is None


def test_cache_eviction(tmp_path):
    from src.llm.serving import ResponseCache
    disk_path = tmp_path / "evict_cache"
    cache = ResponseCache(max_memory=2, disk_path=str(disk_path))
    cache.set("m", "a", "1", temperature=0.0)
    cache.set("m", "b", "2", temperature=0.0)
    cache.set("m", "c", "3", temperature=0.0)
    assert len(cache._memory) <= 2


def test_cache_disk_fallback(tmp_path):
    cache = ResponseCache(max_memory=1, disk_path=str(tmp_path / "df"))
    cache.set("m", "a", "disk_val", temperature=0.0)
    result = cache.get("m", "a", temperature=0.0)
    assert result is not None


def test_cache_ttl(tmp_path):
    cache = ResponseCache(max_memory=10, disk_path=str(tmp_path / "ttl"))
    cache.set("m", "q", "resp", temperature=0.0)
    key = cache._make_key("m", "q", 0.0)
    cache._memory.pop(key, None)
    result = cache.get("m", "q", temperature=0.0)
    assert result is not None


def test_speculative_decoder_creation():
    sd = SpeculativeDecoder(draft_model="tiny", target_model="big")
    assert sd.draft_model == "tiny"
    assert sd.target_model == "big"


def test_continuous_batcher():
    batcher = ContinuousBatcher(batch_window_ms=50)
    assert batcher.batch_window == 0.05
