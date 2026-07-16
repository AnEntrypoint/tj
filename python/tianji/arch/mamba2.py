"""Mamba-2 selective state-space layer (lightweight reference implementation)."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.utils.checkpoint


@dataclass
class MambaConfig:
    dim: int = 16
    state_dim: int = 4
    d_inner: int = 32
    dt_rank: int = 4


class Mamba2Layer(nn.Module):
    def __init__(self, cfg: MambaConfig):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.dim, cfg.d_inner * 2, bias=False)
        self.dt_proj = nn.Linear(cfg.dim, cfg.d_inner, bias=False)
        self.out_proj = nn.Linear(cfg.d_inner, cfg.dim, bias=False)
        # A is stored as log of a positive rate; decay = exp(-dt * exp(A_log)) in (0,1)
        self.A_log = nn.Parameter(torch.zeros(cfg.d_inner, cfg.state_dim))
        self.D = nn.Parameter(torch.ones(cfg.d_inner))

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None):
        if self.training and x.requires_grad:
            return torch.utils.checkpoint.checkpoint(
                self._forward_body, x, state, use_reentrant=True)
        return self._forward_body(x, state)

    def _forward_body(self, x: torch.Tensor, state: torch.Tensor | None = None):
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        dt = self.dt_proj(x).sigmoid()
        A_v = torch.exp(self.A_log)
        D_v = self.D
        log_a = -dt.unsqueeze(-1) * A_v.view(1, 1, A_v.shape[0], A_v.shape[1])
        log_fwd = torch.cumsum(log_a, dim=1).clamp(min=-50.0)
        xin = x_in.unsqueeze(-1)
        terms = xin * torch.exp(-log_fwd)
        s = torch.exp(log_fwd) * torch.cumsum(terms, dim=1)
        if state is not None:
            s = s + torch.exp(log_fwd) * state.unsqueeze(1)
        out = s.sum(dim=-1) + D_v * x_in
        y = out * z.sigmoid()
        y = self.out_proj(y)
        return y, s[:, -1].contiguous()
