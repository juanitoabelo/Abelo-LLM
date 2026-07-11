from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference import load_model, load_tokenizer, generate_text
from src.multimodal import generate_artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Small CLI wrapper for generation")
    parser.add_argument("--prompt", default="hello", help="Prompt to start generation")
    parser.add_argument("--checkpoint", default="checkpoints/tiny_transformer_best.pt", help="Path to the model checkpoint")
    parser.add_argument("--tokenizer", default="checkpoints/tokenizer.json", help="Path to tokenizer file")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=None, help="Restrict sampling to the top-k tokens")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling cutoff")
    parser.add_argument("--mode", choices=["auto", "text", "code", "image", "infographic", "video"], default="auto", help="Mode for multimodal artifact generation")
    parser.add_argument("--output", default=None, help="Optional path for a generated file artifact")
    args = parser.parse_args()

    if args.output:
        output_path = generate_artifact(args.prompt, args.output)
        print(output_path)
        raise SystemExit(0)

    tokenizer = load_tokenizer(args.tokenizer)
    model = load_model(args.checkpoint)
    out = generate_text(args.prompt, tokenizer, model, args.max_new_tokens, args.temperature, args.top_k, args.top_p)
    print(out)
