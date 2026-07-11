"""Multi-Token Prediction head (DeepSeek-V3 style)."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MTPConfig:
    dim: int = 16
    vocab_size: int = 32
    depth: int = 2


class MTPHead(nn.Module):
    def __init__(self, cfg: MTPConfig):
        super().__init__()
        self.cfg = cfg
        self.norms = nn.ModuleList([nn.LayerNorm(cfg.dim) for _ in range(cfg.depth)])
        self.proj = nn.ModuleList([nn.Linear(cfg.dim, cfg.dim, bias=False) for _ in range(cfg.depth)])
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

    def forward(self, x: torch.Tensor):
        outs = []
        h = x
        for d in range(self.cfg.depth):
            h = self.proj[d](self.norms[d](h)) + x
            outs.append(self.head(h))
        return outs

    def speculate(self, x: torch.Tensor):
        logits = self.forward(x)
        return [logits[d].argmax(dim=-1) for d in range(self.cfg.depth)]
