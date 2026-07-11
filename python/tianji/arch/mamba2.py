"""Mamba-2 selective state-space layer (lightweight reference implementation)."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


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

    def forward(self, x: torch.Tensor):
        b, t, _ = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        dt = self.dt_proj(x).sigmoid()  # (b, t, d_inner)
        A = torch.exp(self.A_log)  # (d_inner, state_dim) positive
        state = torch.zeros(b, self.cfg.d_inner, self.cfg.state_dim, device=x.device, dtype=x.dtype)
        y_acc = []
        for ti in range(t):
            decay = torch.exp(-dt[:, ti, :].unsqueeze(2) * A.unsqueeze(0))
            state = decay * state + x_in[:, ti, :].unsqueeze(2)
            out = state.sum(dim=2) + self.D * x_in[:, ti, :]
            y_acc.append(out)
        y = torch.stack(y_acc, dim=1)
        y = y * z.sigmoid()
        y = self.out_proj(y)
        return y, state.transpose(1, 2).contiguous()
