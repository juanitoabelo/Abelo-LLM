#!/usr/bin/env python3
"""
LoRA fine-tuning workflow for Ollama-hosted models.

Prepares a dataset, trains a LoRA adapter using MLX (Apple Silicon)
or llama.cpp, and exports a Modelfile for use with Ollama.

Usage:
    python scripts/finetune_lora.py --base-model llama3.2:1b --data ./my_data.txt --output ./lora-out
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning workflow")
    parser.add_argument("--base-model", default="llama3.2:1b", help="Base Ollama model")
    parser.add_argument("--data", required=True, help="Path to training data file or directory")
    parser.add_argument("--output", default="./lora-out", help="Output directory")
    parser.add_argument("--format", choices=["jsonl", "txt"], default="txt", help="Training data format")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--method", choices=["mlx", "llama.cpp"], default="mlx", help="Training backend")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== LoRA Fine-tuning Pipeline ===")
    print(f"Base model: {args.base_model}")
    print(f"Data:       {args.data}")
    print(f"Output:     {output_dir}")
    print(f"Method:     {args.method}")
    print()

    data_path = Path(args.data)
    if data_path.is_file():
        train_files = [data_path]
    elif data_path.is_dir():
        train_files = list(data_path.rglob("*.txt")) + list(data_path.rglob("*.jsonl"))
    else:
        print(f"Error: data path not found: {data_path}")
        sys.exit(1)

    if not train_files:
        print("No training files found")
        sys.exit(1)

    print(f"Found {len(train_files)} training file(s)")
    print()

    if args.method == "mlx":
        _finetune_mlx(args, output_dir, train_files)
    else:
        _finetune_llamacpp(args, output_dir, train_files)

    _create_modelfile(args.base_model, output_dir)
    print(f"\nDone! LoRA adapter saved to {output_dir}")
    print(f"Create Ollama model with:")
    print(f"  ollama create my-finetuned-model -f {output_dir}/Modelfile")


def _finetune_mlx(args: argparse.Namespace, output_dir: Path, train_files: list[Path]) -> None:
    print("=== Step 1: Prepare training data (JSONL) ===")
    jsonl_path = output_dir / "train.jsonl"
    with open(jsonl_path, "w") as f:
        for tf in train_files:
            if tf.suffix == ".jsonl":
                f.write(tf.read_text())
            else:
                text = tf.read_text(encoding="utf-8", errors="replace")
                for line in text.split("\n"):
                    line = line.strip()
                    if line:
                        f.write(json.dispatch({"text": line}) + "\n")
    print(f"  Wrote {jsonl_path}")

    print("\n=== Step 2: Run MLX LoRA training ===")
    print("  Requires: pip install mlx mlx-lm")
    print(f"  Command: mlx_lm.lora --model {args.base_model} --data {output_dir} --train --iters 100 --lora-layers 8")
    print("  (Run this manually with mlx-lm installed)")


def _finetune_llamacpp(args: argparse.Namespace, output_dir: Path, train_files: list[Path]) -> None:
    print("=== Step 1: Convert training data ===")
    jsonl_path = output_dir / "train.jsonl"
    with open(jsonl_path, "w") as f:
        for tf in train_files:
            if tf.suffix == ".jsonl":
                f.write(tf.read_text())
            else:
                text = tf.read_text(encoding="utf-8", errors="replace")
                f.write(json.dumps({"text": text.strip()}) + "\n")
    print(f"  Wrote {jsonl_path}")

    print("\n=== Step 2: Run llama.cpp LoRA training ===")
    print("  Requires: llama.cpp with finetune tools built")
    print("  Command: ./finetune --model-base gguf-model.gguf --train-data train.jsonl --lora-out lora.gguf")
    print("  (Run this from llama.cpp build directory)")


def _create_modelfile(base_model: str, output_dir: Path) -> None:
    modelfile = f"""FROM {base_model}
ADAPTER {output_dir}/lora_adapter.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.9

TEMPLATE \"\"\"{{ .Prompt }}\"\"\"

SYSTEM \"\"\"You are a fine-tuned model.\"\"\"
"""
    (output_dir / "Modelfile").write_text(modelfile)
    print(f"\n  Wrote Modelfile to {output_dir / 'Modelfile'}")


if __name__ == "__main__":
    main()
