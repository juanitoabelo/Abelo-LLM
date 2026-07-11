from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from src.model import TinyTransformer
from src.tokenizer import BPETokenizer


def load_checkpoint(path: str) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu")


def load_tokenizer(tokenizer_path: str) -> BPETokenizer:
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)
    return tokenizer


def load_model(checkpoint_path: str) -> TinyTransformer:
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["model_config"]
    model = TinyTransformer(
        vocab_size=checkpoint["vocab_size"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        max_context_len=config["block_size"],
        ff_hidden_size=config["ff_hidden_size"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def generate_text(prompt: str, tokenizer: BPETokenizer, model: TinyTransformer, max_new_tokens: int = 8) -> str:
    model.eval()
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        return prompt

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = torch.tensor([token_ids[-model.positional_embedding.num_embeddings :]], dtype=torch.long)
            logits = model(context)
            next_token = torch.argmax(logits[0, -1], dim=-1).item()
            token_ids.append(next_token)

    return tokenizer.decode(token_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with the tiny transformer")
    parser.add_argument("--prompt", default="hello", help="Prompt to start generation")
    parser.add_argument("--checkpoint", default="checkpoints/tiny_transformer.pt", help="Path to the model checkpoint")
    parser.add_argument("--tokenizer", default="checkpoints/tokenizer.json", help="Path to the saved tokenizer JSON")
    parser.add_argument("--max-new-tokens", type=int, default=8, help="Number of tokens to generate")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer)
    model = load_model(args.checkpoint)
    print(generate_text(args.prompt, tokenizer, model, args.max_new_tokens))
