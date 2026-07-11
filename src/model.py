from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_context_len: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.register_buffer("mask", torch.tril(torch.ones(max_context_len, max_context_len)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:seq_len, :seq_len] == 0, float("-inf"))
        attention_weights = F.softmax(scores, dim=-1)

        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(context)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_size)
        self.fc2 = nn.Linear(hidden_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_context_len: int, ff_hidden_size: int) -> None:
        super().__init__()
        self.attn = CausalMultiHeadAttention(d_model, num_heads, max_context_len)
        self.ffn = FeedForward(d_model, ff_hidden_size)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, max_context_len: int, ff_hidden_size: int) -> None:
        super().__init__()
        self.max_context_len = max_context_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Embedding(max_context_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, max_context_len, ff_hidden_size) for _ in range(num_layers)]
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.embedding(token_ids) + self.positional_embedding(positions)

        for block in self.blocks:
            x = block(x)

        return self.lm_head(x)
