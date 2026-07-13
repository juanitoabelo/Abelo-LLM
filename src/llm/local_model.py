from __future__ import annotations

from pathlib import Path
from typing import AsyncGenerator, Optional

from src.config.settings import get_settings


class LocalModelBackend:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        checkpoint_path = Path(self.settings.custom_checkpoint)
        tokenizer_path = Path(self.settings.custom_tokenizer)
        if not checkpoint_path.exists() or not tokenizer_path.exists():
            raise RuntimeError(f"Custom model not found at {checkpoint_path} or {tokenizer_path}")
        from src.model import TinyTransformer
        from src.inference import load_checkpoint, load_tokenizer

        self.tokenizer = load_tokenizer(str(tokenizer_path))
        checkpoint = load_checkpoint(str(checkpoint_path))
        config = checkpoint["model_config"]
        self.model = TinyTransformer(
            vocab_size=checkpoint["vocab_size"],
            d_model=config["d_model"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            max_context_len=config["block_size"],
            ff_hidden_size=config["ff_hidden_size"],
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self._loaded = True

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 128,
        top_k: int = 40,
        top_p: float = 0.95,
    ) -> AsyncGenerator[str, None]:
        self._load()
        import torch
        from src.inference import _generate_tokens

        tokens = _generate_tokens(
            prompt,
            self.tokenizer,
            self.model,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        for token in tokens:
            yield token

    async def is_available(self) -> bool:
        if self._loaded:
            return True
        checkpoint_path = Path(self.settings.custom_checkpoint)
        tokenizer_path = Path(self.settings.custom_tokenizer)
        return checkpoint_path.exists() and tokenizer_path.exists()
