# Phase-by-Phase Implementation Instructions

## 1. Project Objective

Build a small, from-scratch autoregressive language model using PyTorch with no Hugging Face transformers dependency. The project should progress from a minimal local prototype to a larger pretraining setup while staying organized around the provided file structure.

## 2. Chronological Phase-by-Phase Roadmap

### Phase 1: Environment Setup & Project Layout
**Goal:** Configure local and cloud compute instances with essential deep learning libraries.

- Install Ubuntu Linux with updated NVIDIA CUDA Toolkits and drivers.
- Create an isolated Python environment containing:
  - Python
  - PyTorch (`torch`)
  - `numpy`
- Do not use Hugging Face `transformers`.
- Organize the repository as follows:

```text
├── data/               # Raw corpora, clean text samples, and vocabularies
├── src/
│   ├── tokenizer.py    # Byte-Pair Encoding logic
│   ├── model.py        # Neural Network layers (Attention, MLP, Transformer)
│   ├── train.py        # Optimization step, loss calculation, checkpointing
│   └── inference.py    # Text generation engine
└── configs/            # Architecture scales (dimension, layer count, heads)
```

### Phase 2: Tokenization & Data Engineering
**Goal:** Source, cleanse, and partition raw strings into discrete numerical representations.

- Download open-source text repositories such as OpenWebText or localized Wikipedia dumps.
- Build an ingestion pipeline to remove structural markup, corrupted bytes, formatting remnants, and boilerplate text.
- Implement a Byte-Pair Encoding (BPE) tokenizer and train it natively over the text pool.
- Target a vocabulary size of around 32,000 token indices.
- Create a custom data loader that converts tokenized text into training sequences where each input is shifted by one token to form `(X, Y)` pairs.

### Phase 3: Coding the Transformer Layers
**Goal:** Program structural neural architecture blocks using fundamental matrix operations.

- Create embedding tables for token projection and position-aware mappings such as RoPE.
- Implement causal self-attention using scaled dot-product attention.
- Use a lower-triangular mask so the model cannot see future tokens during training or inference.
- Implement feed-forward layers using SwiGLU activation.
- Assemble multiple attention and MLP blocks with residual connections and RMSNorm to preserve signal quality.

### Phase 4: Micro-Scale Testing & Validation
**Goal:** Verify internal logic and training stability over miniature computational scales.

- Build a training loop using AdamW and cross-entropy loss.
- Add simple logging and checkpoint saving for model weights.
- Run a micro-scale model with roughly 10M parameters on local hardware.
- Train on small corpora such as Shakespeare texts for 24–48 hours.
- Test inference with localized prompts to confirm the model produces coherent phrases in the target style.

### Phase 5: Distributed Scaling & Pre-Training
**Goal:** Transition checked codebases onto large enterprise compute setups for full baseline training.

- Integrate PyTorch FSDP or Megatron-LM-style distributed wrappers.
- Use multi-device infrastructure with high-throughput NVIDIA H100-class hardware where possible.
- Scale the model to 1B+ parameters and train over large text datasets.
- Monitor global loss continuously over extended training runs.

### Phase 6: Instruction Tuning & Safety Alignment
**Goal:** Adapt the raw base text completion engine into a structured conversation assistant.

- Perform supervised fine-tuning (SFT) using curated chat-style prompt data.
- Teach the model to follow conversational structure and instruction formatting.
- Apply safety alignment using Direct Preference Optimization (DPO) or similar methods with ranked answer pairs.
- Reduce biased, toxic, or dangerous outputs through preference-based training.

## 3. Reference Implementation: Causal Multi-Head Attention Block

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_context_len):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.out_proj = nn.Linear(d_model, d_model)

        self.register_buffer("mask", torch.tril(torch.ones(max_context_len, max_context_len)))

    def forward(self, x):
        B, T, C = x.shape

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        attention_weights = F.softmax(scores, dim=-1)

        context_out = torch.matmul(attention_weights, v)
        context_out = context_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(context_out)
```

## 4. Practical Notes for This Repository

- Implement the tokenizer in [src/tokenizer.py](src/tokenizer.py).
- Implement the model components in [src/model.py](src/model.py).
- Implement the training loop in [src/train.py](src/train.py).
- Implement generation logic in [src/inference.py](src/inference.py).
- Keep configuration values such as hidden size, number of layers, and attention heads in [configs](configs).
