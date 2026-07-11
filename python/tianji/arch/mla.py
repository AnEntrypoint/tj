"""Multi-head Latent Attention (DeepSeek-style MLA) with low-rank KV compression."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MLAConfig:
    dim: int = 16
    n_heads: int = 2
    head_dim: int = 8
    kv_latent: int = 8


class MLALayer(nn.Module):
    def __init__(self, cfg: MLAConfig):
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.kv_down = nn.Linear(cfg.dim, cfg.kv_latent, bias=False)
        self.kv_up = nn.Linear(cfg.kv_latent, cfg.n_heads * cfg.head_dim * 2, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.cfg.n_heads, self.cfg.head_dim)
        c = self.kv_down(x)
        kv = self.kv_up(c).view(b, t, self.cfg.n_heads, self.cfg.head_dim * 2)
        k, v = kv.chunk(2, dim=-1)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(b, t, -1)
        out = self.o_proj(out)
        return out, c
