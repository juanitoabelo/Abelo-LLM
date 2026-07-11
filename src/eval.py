from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import TextDataset, clean_text
from src.tokenizer import BPETokenizer, SentencePieceTokenizer
from src.model import TinyTransformer


def load_checkpoint(path: str):
    return torch.load(path, map_location="cpu")


def compute_perplexity(checkpoint_path: str, tokenizer_path: str | None, texts: list[str], block_size: int):
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["model_config"]
    vocab_size = checkpoint.get("vocab_size")

    # load tokenizer
    tokenizer = None
    if tokenizer_path:
        try:
            tokenizer = BPETokenizer()
            tokenizer.load(tokenizer_path)
        except Exception:
            if SentencePieceTokenizer is not None:
                sp = SentencePieceTokenizer()
                sp.load(tokenizer_path)
                tokenizer = sp
    if tokenizer is None:
        raise RuntimeError("Unable to load tokenizer")

    model = TinyTransformer(
        vocab_size=vocab_size,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        max_context_len=config["block_size"],
        ff_hidden_size=config.get("ff_hidden_size", config["d_model"] * 4),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = TextDataset.from_texts(texts, tokenizer, block_size=block_size)
    loader = DataLoader(dataset, batch_size=8)

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
            total_loss += loss.item() * xb.size(0) * xb.size(1)
            total_tokens += xb.size(0) * xb.size(1)

    avg_nll = total_loss / max(1, total_tokens)
    ppl = math.exp(avg_nll)
    return ppl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model perplexity on held-out texts")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--texts", nargs="+", required=True)
    parser.add_argument("--block-size", type=int, default=128)
    args = parser.parse_args()

    ppl = compute_perplexity(args.checkpoint, args.tokenizer, args.texts, args.block_size)
    print(f"Perplexity: {ppl:.3f}")
