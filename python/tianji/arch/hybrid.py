"""Hybrid stack: interleaved Mamba-2 and MLA+MoE blocks (DeepSeek-style hybrid)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn

from .mamba2 import Mamba2Layer, MambaConfig
from .mla import MLALayer, MLAConfig
from .moe import MoELayer, MoEConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight


@dataclass
class HybridConfig:
    dim: int = 16
    n_layers: int = 27
    state_dim: int = 4
    d_inner: int = 32
    dt_rank: int = 4
    n_heads: int = 2
    head_dim: int = 8
    kv_latent: int = 8
    n_experts: int = 4
    n_active: int = 2
    shared_experts: int = 1
    expert_hidden: int = 32


class MambaBlock(nn.Module):
    def __init__(self, cfg: HybridConfig):
        super().__init__()
        self.norm = RMSNorm(cfg.dim)
        self.mamba = Mamba2Layer(MambaConfig(dim=cfg.dim, state_dim=cfg.state_dim,
                                             d_inner=cfg.d_inner, dt_rank=cfg.dt_rank))

    def forward(self, x):
        y, _ = self.mamba(self.norm(x))
        return x + y, torch.zeros((), device=x.device, dtype=x.dtype)


class MLAMoELayer(nn.Module):
    def __init__(self, cfg: HybridConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.dim)
        self.mla = MLALayer(MLAConfig(dim=cfg.dim, n_heads=cfg.n_heads,
                                      head_dim=cfg.head_dim, kv_latent=cfg.kv_latent))
        self.norm2 = RMSNorm(cfg.dim)
        self.moe = MoELayer(MoEConfig(dim=cfg.dim, n_experts=cfg.n_experts, n_active=cfg.n_active,
                                      shared_experts=cfg.shared_experts, expert_hidden=cfg.expert_hidden))

    def forward(self, x):
        a, _ = self.mla(self.norm1(x))
        h = x + a
        m, aux = self.moe(self.norm2(h))
        return h + m, aux


def _build_layers(cfg: HybridConfig) -> List[nn.Module]:
    layers: List[nn.Module] = []
    for i in range(cfg.n_layers):
        if i % 3 == 2:
            layers.append(MLAMoELayer(cfg))
        else:
            layers.append(MambaBlock(cfg))
    return layers


class HybridStack(nn.Module):
    def __init__(self, cfg: HybridConfig):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList(_build_layers(cfg))
        self.norm = RMSNorm(cfg.dim)

    def forward(self, x: torch.Tensor):
        aux_sum = torch.zeros((), device=x.device, dtype=x.dtype)
        for layer in self.layers:
            x, aux = layer(x)
            aux_sum = aux_sum + aux
        return self.norm(x), aux_sum
