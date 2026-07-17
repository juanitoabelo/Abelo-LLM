"""End-to-end training pipeline for the custom TinyTransformer.

Usage:
  python scripts/train_pipeline.py --dataset data/tiny_shakespeare --model-size small
  python scripts/train_pipeline.py --dataset data/tiny_stories --model-size base --epochs 20
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train import train_model, load_texts_from_dir
from src.tokenizer import BPETokenizer, HFTokenizersWrapper
from configs.model_config import MODEL_CONFIGS


def prepare_tokenizer(texts: list[str], vocab_size: int, output_dir: str) -> None:
    """Train and save a tokenizer separately so we can inspect it."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok_path = out / "tokenizer.json"

    if HFTokenizersWrapper is not None:
        tok = HFTokenizersWrapper(vocab_size=vocab_size)
        tok.fit(texts)
        tok.save(tok_path)
        print(f"Tokenizer saved: {tok_path} (vocab_size={tok.vocab_size})")
    else:
        tok = BPETokenizer(target_vocab_size=vocab_size)
        tok.fit(texts)
        tok.save(tok_path)
        print(f"Tokenizer saved: {tok_path} (vocab_size={tok.vocab_size})")


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end custom LLM training pipeline")
    parser.add_argument("--dataset", required=True, help="Path to training data directory or file")
    parser.add_argument("--model-size", default="small", choices=list(MODEL_CONFIGS), help="Model configuration")
    parser.add_argument("--vocab-size", type=int, default=2048, help="Vocabulary size for tokenizer")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--output-dir", default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--tokenizer-only", action="store_true", help="Only train and save the tokenizer")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        print("Run 'python scripts/download_data.py' first to get sample datasets.")
        sys.exit(1)

    if dataset_path.is_file():
        texts = [dataset_path.read_text(encoding="utf-8", errors="ignore")]
    else:
        texts = load_texts_from_dir(str(dataset_path))

    print(f"Loaded {len(texts)} text(s), total chars: {sum(len(t) for t in texts):,}")
    if not texts:
        print("No training texts found!")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining tokenizer (vocab_size={args.vocab_size})...")
    prepare_tokenizer(texts, args.vocab_size, str(output_dir))

    if args.tokenizer_only:
        print("Tokenizer training complete.")
        return

    config = MODEL_CONFIGS[args.model_size]
    param_count = (
        config["d_model"] * args.vocab_size  # embedding
        + config["max_context_len"] * config["d_model"]  # pos emb
        + config["num_layers"] * (
            4 * config["d_model"] * config["d_model"]  # QKV + out
            + config["ff_hidden_size"] * config["d_model"] * 2  # SwiGLU 2 fcs
        )
        + config["d_model"] * args.vocab_size  # lm_head
    )

    print(f"\n{'='*50}")
    print(f"Model: {args.model_size}")
    print(f"Config: d_model={config['d_model']}, layers={config['num_layers']}, heads={config['num_heads']}")
    print(f"Parameters: ~{param_count / 1e6:.1f}M")
    print(f"Block size: {config['block_size']}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.learning_rate}")
    print(f"{'='*50}\n")

    train_model(
        texts=texts,
        output_dir=str(output_dir),
        model_size=args.model_size,
        target_vocab_size=args.vocab_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        tokenizer_choice="auto",
        use_amp=False,
    )

    print(f"\nTraining complete! Checkpoints saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
