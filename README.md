# my_custom_llm

Small from-scratch autoregressive language model (toy) using PyTorch.

Quick start

1. Create a Python environment and install requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Train on a few sample phrases:

```bash
python3 src/train.py --texts "hello world" "hello there" "world hello" --output-dir checkpoints --epochs 3
```

3. Generate text:

```bash
python3 src/cli.py --prompt "hello" --checkpoint checkpoints/tiny_transformer_best.pt --tokenizer checkpoints/tokenizer.json --max-new-tokens 16
```

4. Optional: enable the faster tokenizer backend and TensorBoard logs:

```bash
python3 src/train.py --texts "hello world" "hello there" "world hello" --output-dir checkpoints --tokenizer-choice tokenizers
```

TensorBoard logs will be written to `checkpoints/tensorboard/` if `tensorboard` is installed.

Files

- `src/tokenizer.py` - toy BPE and optional `SentencePieceTokenizer` wrapper.
- `src/model.py` - transformer building blocks and `TinyTransformer`.
- `src/train.py` - training loop with checkpointing and simple validation.
- `src/inference.py` - model loading and greedy generation.
- `src/eval.py` - compute perplexity on small held-out texts.
- `configs/model_config.py` - model size presets.

Notes

- This project is intended as a learning scaffold. For production you should use robust tokenizers, larger datasets, distributed training, and careful evaluation.
