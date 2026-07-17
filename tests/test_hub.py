"""Tests for model hub integration."""

import pytest
from src.llm.hub import ModelHub


@pytest.fixture
def hub():
    return ModelHub()


@pytest.mark.asyncio
async def test_suggest_models_chat(hub):
    models = await hub.suggest_models("chat")
    assert len(models) > 0
    assert any("llama" in m["id"] for m in models)


@pytest.mark.asyncio
async def test_suggest_models_code(hub):
    models = await hub.suggest_models("code")
    assert len(models) > 0
    assert any("code" in m["id"] or "deepseek" in m["id"] for m in models)


@pytest.mark.asyncio
async def test_suggest_models_vision(hub):
    models = await hub.suggest_models("vision")
    assert len(models) > 0


@pytest.mark.asyncio
async def test_suggest_models_unknown(hub):
    models = await hub.suggest_models("unknown_task")
    assert len(models) > 0
