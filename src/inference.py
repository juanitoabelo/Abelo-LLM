from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from src.model import TinyTransformer
from src.tokenizer import BPETokenizer, SentencePieceTokenizer, HFTokenizersWrapper


def load_checkpoint(path: str) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu")


def load_tokenizer(tokenizer_path: str):
    path = Path(tokenizer_path)
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    if path.suffix == ".json":
        tokenizer = BPETokenizer()
        tokenizer.load(path)
        return tokenizer

    if path.suffix == ".model" and SentencePieceTokenizer is not None:
        tokenizer = SentencePieceTokenizer(model_prefix=str(path.with_suffix("")))
        tokenizer.load(path)
        return tokenizer

    if path.suffix == ".json" and HFTokenizersWrapper is not None:
        tokenizer = HFTokenizersWrapper()
        tokenizer.load(path)
        return tokenizer

    raise ValueError(f"Unsupported tokenizer format: {tokenizer_path}")


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


def generate_text(
    prompt: str,
    tokenizer,
    model: TinyTransformer,
    max_new_tokens: int = 8,
    temperature: float = 0.8,
    top_k: int | None = None,
    top_p: float = 1.0,
) -> str:
    tokens = list(_generate_tokens(prompt, tokenizer, model, max_new_tokens, temperature, top_k, top_p))
    return "".join(tokens)


def generate_text_stream(
    prompt: str,
    tokenizer,
    model: TinyTransformer,
    max_new_tokens: int = 8,
    temperature: float = 0.8,
    top_k: int | None = None,
    top_p: float = 1.0,
) -> list[str]:
    return list(_generate_tokens(prompt, tokenizer, model, max_new_tokens, temperature, top_k, top_p))


def _generate_tokens(
    prompt: str,
    tokenizer,
    model: TinyTransformer,
    max_new_tokens: int = 8,
    temperature: float = 0.8,
    top_k: int | None = None,
    top_p: float = 1.0,
):
    model.eval()
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        yield prompt
        return

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = torch.tensor([token_ids[-model.max_context_len :]], dtype=torch.long)
            logits = model(context)
            next_token_logits = logits[0, -1] / max(temperature, 1e-6)

            if top_k is not None:
                top_k = min(top_k, next_token_logits.size(-1))
                top_logits, top_indices = torch.topk(next_token_logits, k=top_k)
                filtered = torch.full_like(next_token_logits, float("-inf"))
                filtered[top_indices] = top_logits
                next_token_logits = filtered

            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float("-inf")

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            token_ids.append(next_token)
            decoded = tokenizer.decode([next_token])
            if decoded:
                yield decoded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with the tiny transformer")
    parser.add_argument("--prompt", default="hello", help="Prompt to start generation")
    parser.add_argument("--checkpoint", default="checkpoints/tiny_transformer.pt", help="Path to the model checkpoint")
    parser.add_argument("--tokenizer", default="checkpoints/tokenizer.json", help="Path to the saved tokenizer JSON")
    parser.add_argument("--max-new-tokens", type=int, default=8, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=None, help="Restrict sampling to the top-k tokens")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling cutoff")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer)
    model = load_model(args.checkpoint)
    print(generate_text(args.prompt, tokenizer, model, args.max_new_tokens, args.temperature, args.top_k, args.top_p))
