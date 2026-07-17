"""Tests for multi-provider smart routing."""

import pytest
from src.llm.provider_router import ProviderRouter


@pytest.fixture
def router():
    return ProviderRouter()


def test_task_classification_code(router):
    assert router.classify_task("Write a Python function to sort a list") == "code"


def test_task_classification_math(router):
    assert router.classify_task("Calculate 15 + 27") == "math"


def test_task_classification_simple(router):
    assert router.classify_task("hello") == "simple"


def test_task_classification_general(router):
    assert router.classify_task("Tell me about the history of Rome") == "general"


def test_recommend_model_code(router):
    available = ["llama3.2:1b", "deepseek-coder:latest", "gemma4:latest"]
    assert router.recommend_model("Write code", available) == "deepseek-coder:latest"


def test_recommend_model_simple(router):
    available = ["llama3.2:1b", "qwen3.5:latest"]
    assert router.recommend_model("hi", available) == "llama3.2:1b"


def test_recommend_model_fallback(router):
    assert router.recommend_model("test", []) == "llama3.2:1b"


def test_select_backend_ollama(router):
    assert router.select_backend("hello world", custom_model_available=False) == "ollama"
