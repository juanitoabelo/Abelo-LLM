#!/usr/bin/env python3
"""
LoRA fine-tuning workflow using the training infrastructure.

Usage:
    python scripts/finetune_lora.py --base-model llama3.2:1b --data ./my_data.txt --output ./lora-out
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.training.lora import LoRATrainer
from src.training.data import DatasetBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning workflow")
    parser.add_argument("--base-model", default="llama3.2:1b", help="Base Ollama model")
    parser.add_argument("--data", required=True, help="Path to training data file or directory")
    parser.add_argument("--output", default="./lora-out", help="Output directory")
    parser.add_argument("--format", choices=["jsonl", "txt", "md"], default="txt", help="Training data format")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--method", choices=["mlx", "llama.cpp", "unsloth"], default="mlx", help="Training backend")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== LoRA Fine-tuning Pipeline ===")
    print(f"Base model: {args.base_model}")
    print(f"Data:       {args.data}")
    print(f"Output:     {output_dir}")
    print(f"Method:     {args.method}")

    print("\n=== Step 1: Build dataset ===")
    builder = DatasetBuilder(output_dir=output_dir)
    data_path = Path(args.data)

    if data_path.is_file():
        if args.format == "jsonl":
            samples = builder.from_jsonl(data_path)
        else:
            samples = builder.from_text_files(str(data_path.parent), format="raw")
            samples = [s for s in samples if data_path.name in s.get("source", "")]
    elif data_path.is_dir():
        samples = builder.from_text_files(str(data_path), format="raw")
    else:
        print(f"Error: data path not found: {data_path}")
        sys.exit(1)

    if not samples:
        print("No training samples found")
        sys.exit(1)

    jsonl_path = builder.convert_to_jsonl(samples, "train")
    print(f"  {len(samples)} samples -> {jsonl_path}")

    print(f"\n=== Step 2: Train with {args.method} ===")
    trainer = LoRATrainer(backend=args.method)
    result = trainer.train(
        base_model=args.base_model,
        data_path=jsonl_path,
        output_dir=output_dir,
        lora_rank=args.rank,
        epochs=args.epochs,
        learning_rate=args.lr,
    )
    print(f"  Status: {result['status']}")
    print(f"  {result.get('instruction', '')}")

    print(f"\n=== Step 3: Export Modelfile ===")
    adapter_path = output_dir / "lora_adapter.gguf"
    if adapter_path.exists():
        modelfile = trainer.export_modelfile(args.base_model, adapter_path, output_dir / "Modelfile")
        print(f"  Modelfile -> {modelfile}")
        print(f"\nCreate Ollama model with:")
        print(f"  ollama create my-finetuned-model -f {output_dir}/Modelfile")
    else:
        print("  Skipped (adapter not found - run training command above first)")

    print(f"\nDone! LoRA artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
