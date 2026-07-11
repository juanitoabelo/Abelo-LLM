from __future__ import annotations

import pytest

from src.config.settings import get_settings
from src.llm.router import LLMRouter


class TestSettings:
    def test_default_values(self) -> None:
        settings = get_settings()
        assert settings.default_model == "qwen3.5:latest"
        assert settings.temperature == 0.7
        assert settings.max_tokens == 2048
        assert settings.server_port == 8000

    def test_available_models_list(self) -> None:
        settings = get_settings()
        assert len(settings.available_remote_models) > 0
        assert "gemma4:latest" in settings.available_remote_models


class TestRouter:
    @pytest.mark.asyncio
    async def test_backend_check(self) -> None:
        router = LLMRouter()
        backends = await router.check_backends()
        assert "ollama" in backends

    @pytest.mark.asyncio
    async def test_format_chat_messages_with_system(self) -> None:
        router = LLMRouter()
        messages = router.format_chat_messages(
            system_prompt="You are a helpful assistant.",
            history=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
            user_message="how are you?",
        )
        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "how are you?"

    @pytest.mark.asyncio
    async def test_format_chat_messages_without_system(self) -> None:
        router = LLMRouter()
        messages = router.format_chat_messages(
            system_prompt=None,
            history=[],
            user_message="hello",
        )
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_format_chat_messages_without_history(self) -> None:
        router = LLMRouter()
        messages = router.format_chat_messages(
            system_prompt="Be concise.",
            history=[],
            user_message="tell me a joke",
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
