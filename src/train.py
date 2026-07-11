from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch import nn
from torch.utils.data import DataLoader
import logging

from configs.model_config import MODEL_CONFIGS
from src.dataset import TextDataset, clean_text
from src.model import TinyTransformer
from src.tokenizer import BPETokenizer, SentencePieceTokenizer, HFTokenizersWrapper
try:
    # tensorboard is optional
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


def load_texts_from_dir(folder: str) -> List[str]:
    root = Path(folder)
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {folder}")

    texts: List[str] = []
    for file_path in sorted(root.glob("**/*.*")):
        if file_path.is_file() and file_path.suffix.lower() in {".txt", ".md", ".text"}:
            raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
            cleaned = clean_text(raw_text)
            if cleaned:
                texts.append(cleaned)
    return texts


def train_model(
    texts: List[str],
    output_dir: str = "checkpoints",
    model_size: str = "micro",
    target_vocab_size: int = 512,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    tokenizer_choice: str = "auto",
    gradient_clip: float = 1.0,
    use_amp: bool = True,
    gradient_accumulation_steps: int = 1,
    weight_decay: float = 0.01,
) -> None:
    model_config = MODEL_CONFIGS.get(model_size)
    if model_config is None:
        raise ValueError(f"Unknown model size: {model_size}")

    # tokenizer selection: prefer HF tokenizers, then SentencePiece, then BPETokenizer
    if tokenizer_choice in ("auto", "tokenizers") and HFTokenizersWrapper is not None:
        tokenizer = HFTokenizersWrapper(vocab_size=target_vocab_size)
        tokenizer.fit(texts)
    elif tokenizer_choice in ("auto", "sentencepiece") and SentencePieceTokenizer is not None:
        tokenizer = SentencePieceTokenizer(model_prefix="checkpoints/spm", vocab_size=target_vocab_size)
        tokenizer.fit(texts)
    else:
        if tokenizer_choice == "sentencepiece" and SentencePieceTokenizer is None:
            logging.warning("SentencePiece not available; falling back to BPETokenizer")
        if tokenizer_choice == "tokenizers" and HFTokenizersWrapper is None:
            logging.warning("tokenizers library not available; falling back to BPETokenizer")
        tokenizer = BPETokenizer(target_vocab_size=target_vocab_size)
        tokenizer.fit(texts)

    # build dataset
    dataset = TextDataset.from_texts(texts, tokenizer, block_size=model_config["block_size"], stride=1)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=model_config["d_model"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        max_context_len=model_config["block_size"],
        ff_hidden_size=model_config["ff_hidden_size"],
        dropout=model_config.get("dropout", 0.1),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    # training loop with per-epoch checkpointing, TensorBoard (optional), and simple logging
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename=str(output_path / "train.log"), filemode="a", format="%(asctime)s %(levelname)s %(message)s")
    logging.info("Starting training: model_size=%s epochs=%d", model_size, epochs)

    # TensorBoard writer
    tb_writer = None
    if SummaryWriter is not None:
        try:
            tb_writer = SummaryWriter(log_dir=str(output_path / "tensorboard"))
        except Exception:
            tb_writer = None

    best_val_loss = float("inf")
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        model.train()
        for step, (x_batch, y_batch) in enumerate(dataloader, start=1):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32, enabled=use_amp and device.type == "cuda"):
                logits = model(x_batch)
                loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1)) / gradient_accumulation_steps

            if device.type == "cuda" and use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % gradient_accumulation_steps == 0 or step == len(dataloader):
                if device.type == "cuda" and use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                if device.type == "cuda" and use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

            total_loss += loss.item() * gradient_accumulation_steps

        average_loss = total_loss / max(1, len(dataloader))
        logging.info("Epoch %d/%d - train_loss: %.4f", epoch + 1, epochs, average_loss)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {average_loss:.4f}")
        if tb_writer is not None:
            tb_writer.add_scalar("train/loss", average_loss, epoch + 1)

        # save epoch checkpoint
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "vocab_size": tokenizer.vocab_size,
            "epoch": epoch + 1,
        }
        torch.save(checkpoint, output_path / f"tiny_transformer_epoch{epoch+1}.pt")

        # write tokenizer and metadata
        try:
            tokenizer.save(output_path / "tokenizer.json")
        except Exception:
            # some tokenizers may not support save in this wrapper
            pass
        metadata = {"model_size": model_size, "target_vocab_size": target_vocab_size, "vocab_size": getattr(tokenizer, "vocab_size", None)}
        Path(output_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        # optional validation step: small held-out split
        # use final portion of texts as quick validation
        val_texts = texts[-max(1, len(texts) // 10) :]
        if val_texts:
            val_dataset = TextDataset.from_texts(val_texts, tokenizer, block_size=model_config["block_size"], stride=1)
            if len(val_dataset) > 0:
                val_loader = DataLoader(val_dataset, batch_size=batch_size)
                val_loss = 0.0
                model.eval()
                with torch.no_grad():
                    for xb, yb in val_loader:
                        logits = model(xb)
                        l = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
                        val_loss += l.item()
                val_loss = val_loss / max(1, len(val_loader))
                logging.info("Epoch %d validation loss: %.4f", epoch + 1, val_loss)
                model.train()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(checkpoint, output_path / "tiny_transformer_best.pt")
                    if tb_writer is not None:
                        tb_writer.add_scalar("val/loss", val_loss, epoch + 1)

        scheduler.step()

    if tb_writer is not None:
        try:
            tb_writer.flush()
            tb_writer.close()
        except Exception:
            pass

    print(f"Saved checkpoints to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a tiny transformer from text files or sample phrases")
    parser.add_argument("--data-dir", default=None, help="Directory containing training text files")
    parser.add_argument("--texts", nargs="+", default=["hello world", "hello there", "world hello"], help="Sample texts to train on when no data-dir is provided")
    parser.add_argument("--output-dir", default="checkpoints", help="Directory for saved checkpoints")
    parser.add_argument("--model-size", default="micro", choices=list(MODEL_CONFIGS), help="Model size configuration")
    parser.add_argument("--target-vocab-size", type=int, default=512, help="Target vocabulary size for the BPE tokenizer")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--tokenizer-choice", choices=["auto", "sentencepiece", "bpe", "tokenizers"], default="auto", help="Which tokenizer implementation to use")
    parser.add_argument("--gradient-clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Number of optimizer steps to accumulate before updating")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay for AdamW")
    args = parser.parse_args()

    texts = []
    if args.data_dir:
        texts = load_texts_from_dir(args.data_dir)
    else:
        texts = [clean_text(text) for text in args.texts if clean_text(text)]

    if not texts:
        raise ValueError("No training texts were provided")

    # expose tokenizer_choice for train_model
    tokenizer_choice = args.tokenizer_choice

    train_model(
        texts,
        output_dir=args.output_dir,
        model_size=args.model_size,
        target_vocab_size=args.target_vocab_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        tokenizer_choice=args.tokenizer_choice,
        gradient_clip=args.gradient_clip,
        use_amp=not args.no_amp,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
    )
