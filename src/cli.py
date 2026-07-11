from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference import load_model, load_tokenizer, generate_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Small CLI wrapper for generation")
    parser.add_argument("--prompt", default="hello", help="Prompt to start generation")
    parser.add_argument("--checkpoint", default="checkpoints/tiny_transformer_best.pt", help="Path to the model checkpoint")
    parser.add_argument("--tokenizer", default="checkpoints/tokenizer.json", help="Path to tokenizer file")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Number of tokens to generate")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer)
    model = load_model(args.checkpoint)
    out = generate_text(args.prompt, tokenizer, model, args.max_new_tokens)
    print(out)
