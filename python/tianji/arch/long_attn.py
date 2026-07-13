"""Long-context attention: RoPE + memory-efficient global causal attention.

Replaces MLA for the 200k-token requirement. Uses
``F.scaled_dot_product_attention(is_causal=True)`` which dispatches to the
flash / memory-efficient kernel on Ampere (RTX 3060), so a 200k-token window
uses O(n) VRAM instead of materializing an n x n score matrix. RoPE with a
large base keeps positional signal meaningful across 200k+ positions.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LongAttnConfig:
    dim: int = 16
    n_heads: int = 2
    head_dim: int = 8
    rope_base: int = 1_000_000  # large base so rotations stay distinct at 200k+


class LongAttentionLayer(nn.Module):
    def __init__(self, cfg: LongAttnConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.n_heads * cfg.head_dim
        self.q_proj = nn.Linear(cfg.dim, h, bias=False)
        self.k_proj = nn.Linear(cfg.dim, h, bias=False)
        self.v_proj = nn.Linear(cfg.dim, h, bias=False)
        self.o_proj = nn.Linear(h, cfg.dim, bias=False)
        self._freq_cache: dict[int, torch.Tensor] = {}

    def _rope(self, x: torch.Tensor, t: int) -> torch.Tensor:
        # x: (b, h, t, d)
        d = x.size(-1)
        half = d // 2
        if t not in self._freq_cache or self._freq_cache[t].device != x.device:
            inv = 1.0 / (self.cfg.rope_base ** (
                torch.arange(0, d, 2, device=x.device, dtype=x.dtype) / d))
            pos = torch.arange(t, device=x.device, dtype=x.dtype)
            freqs = pos.unsqueeze(-1) * inv.unsqueeze(0)  # (t, half)
            emb = torch.cat([freqs, freqs], dim=-1)        # (t, d)
            self._freq_cache[t] = emb
        emb = self._freq_cache[t]
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        x1 = x[..., :half]
        x2 = x[..., half:]
        rot = torch.cat([-x2, x1], dim=-1)
        return x * cos + rot * sin

    def forward(self, x: torch.Tensor):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2)
        q = self._rope(q, t)
        k = self._rope(k, t)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(b, t, -1)
        out = self.o_proj(out)
        return out, None
