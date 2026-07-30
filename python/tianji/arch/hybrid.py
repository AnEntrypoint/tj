"""Hybrid stack: interleaved Mamba-2 and long-context attention + MoE blocks.

The attention block was upgraded from MLA to ``LongAttentionLayer`` (RoPE +
memory-efficient global causal SDPA) so the model can train and serve on
~200k-token contexts on a 6GB GPU: SDPA with ``is_causal`` dispatches to the
flash / memory-efficient kernel on Ampere, keeping VRAM O(n) instead of
materializing an n x n score matrix. Mamba-2 carries cross-chunk context at
inference via its constant recurrent state.

State convention (tuple-based for CUDA graph compatibility):
    ``state_tuple[j]`` → the Mamba-2 state tensor for the j-th MambaBlock
    in layer order, i.e. the layer at ``self.mamba_block_indices[j]``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .mamba2 import Mamba2Layer, MambaConfig
from .long_attn import LongAttentionLayer, LongAttnConfig
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
    quantize: bool = False  # int4 quantization for MoE + projections


class MambaBlock(nn.Module):
    def __init__(self, cfg: HybridConfig):
        super().__init__()
        self.norm = RMSNorm(cfg.dim)
        self.mamba = Mamba2Layer(MambaConfig(dim=cfg.dim, state_dim=cfg.state_dim,
                                             d_inner=cfg.d_inner, dt_rank=cfg.dt_rank))

    def forward(self, x, state=None):
        y, ns = self.mamba(self.norm(x), state)
        return x + y, torch.zeros((), device=x.device, dtype=x.dtype), ns


class MLAMoELayer(nn.Module):
    def __init__(self, cfg: HybridConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.dim)
        self.attn = LongAttentionLayer(LongAttnConfig(dim=cfg.dim, n_heads=cfg.n_heads,
                                                      head_dim=cfg.head_dim))
        self.norm2 = RMSNorm(cfg.dim)
        self.moe = MoELayer(MoEConfig(dim=cfg.dim, n_experts=cfg.n_experts, n_active=cfg.n_active,
                                      shared_experts=cfg.shared_experts, expert_hidden=cfg.expert_hidden),
                            quantize=cfg.quantize)

    def forward(self, x, state=None, router_bias=None):
        a, _ = self.attn(self.norm1(x))
        h = x + a
        m, aux = self.moe(self.norm2(h), router_bias=router_bias)
        return h + m, aux, None


def _build_layers(cfg: HybridConfig) -> Tuple[List[nn.Module], List[int]]:
    layers: List[nn.Module] = []
    mamba_indices: List[int] = []
    for i in range(cfg.n_layers):
        if i % 3 == 2:
            layers.append(MLAMoELayer(cfg))
        else:
            layers.append(MambaBlock(cfg))
            mamba_indices.append(i)
    return layers, mamba_indices


class HybridStack(nn.Module):
    def __init__(self, cfg: HybridConfig):
        super().__init__()
        self.cfg = cfg
        layers_list, self.mamba_block_indices = _build_layers(cfg)
        self.layers = nn.ModuleList(layers_list)
        self.norm = RMSNorm(cfg.dim)

    def forward(self, x: torch.Tensor,
                state_tuple: Optional[Tuple[torch.Tensor, ...]] = None,
                router_bias: Optional[torch.Tensor] = None):
        aux_sum = torch.zeros((), device=x.device, dtype=x.dtype)
        next_tensors: List[torch.Tensor] = []
        ti = 0
        for i, layer in enumerate(self.layers):
            if isinstance(layer, MambaBlock):
                s = state_tuple[ti] if state_tuple is not None else None
                x, aux, ns = layer(x, s)
                if ns is not None:
                    next_tensors.append(ns)
                ti += 1
            else:
                x, aux, _ = layer(x, router_bias=router_bias)
            aux_sum = aux_sum + aux
        return self.norm(x), aux_sum, tuple(next_tensors) if next_tensors else None

    @property
    def num_stateful(self) -> int:
        return len(self.mamba_block_indices)
