import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.model_config import MODEL_CONFIGS
from src.dataset import TextDataset
from src.model import TinyTransformer
from src.tokenizer import BPETokenizer


def test_bpe_tokenizer_round_trip():
    tokenizer = BPETokenizer(target_vocab_size=64)
    tokenizer.fit(["hello world", "hello there", "world hello"])

    tokens = tokenizer.encode("hello world")
    decoded = tokenizer.decode(tokens)

    assert decoded == "hello world"
    assert len(tokens) > 0
    assert tokenizer.vocab_size > 1


def test_text_dataset_build():
    tokenizer = BPETokenizer(target_vocab_size=64)
    training_texts = ["hello world", "hello there"] * 16
    tokenizer.fit(training_texts)
    dataset = TextDataset.from_texts(training_texts, tokenizer, block_size=8, stride=1)

    assert len(dataset) > 0
    x, y = dataset[0]
    assert x.shape == (8,)
    assert y.shape == (8,)


def test_tiny_transformer_forward_shape():
    config = MODEL_CONFIGS["micro"]
    model = TinyTransformer(
        vocab_size=128,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        max_context_len=config["block_size"],
        ff_hidden_size=config["ff_hidden_size"],
    )
    x = torch.randint(0, 128, (2, 8))
    logits = model(x)

    assert logits.shape == (2, 8, 128)
