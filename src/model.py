from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    if x.size(-1) % 2 != 0:
        return x
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_context_len: int, use_rotary: bool = True) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_rotary = use_rotary

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.register_buffer("mask", torch.tril(torch.ones(max_context_len, max_context_len)))

    def _build_rotary_embedding(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self.head_dim < 2 or not self.use_rotary:
            return torch.ones(1, 1, seq_len, 1, device=device), torch.zeros(1, 1, seq_len, 1, device=device)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim))
        positions = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(-1)
        freqs = positions * inv_freq
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)
        return cos, sin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_rotary and self.head_dim >= 2:
            cos, sin = self._build_rotary_embedding(seq_len, q.device)
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:seq_len, :seq_len] == 0, float("-inf"))
        attention_weights = F.softmax(scores, dim=-1)

        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(context)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_size)
        self.fc_gate = nn.Linear(d_model, hidden_size)
        self.fc2 = nn.Linear(hidden_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.fc_gate(x))
        return self.fc2(gate * self.fc1(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_context_len: int, ff_hidden_size: int, use_rotary: bool = True) -> None:
        super().__init__()
        self.attn = CausalMultiHeadAttention(d_model, num_heads, max_context_len, use_rotary=use_rotary)
        self.ffn = SwiGLU(d_model, ff_hidden_size)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        max_context_len: int,
        ff_hidden_size: int,
        dropout: float = 0.1,
        use_rotary: bool = True,
    ) -> None:
        super().__init__()
        self.max_context_len = max_context_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Embedding(max_context_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, max_context_len, ff_hidden_size, use_rotary=use_rotary) for _ in range(num_layers)]
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.embedding(token_ids) + self.positional_embedding(positions)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        return self.lm_head(x)
