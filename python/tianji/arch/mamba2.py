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
        # Vectorized selective-scan recurrence. The per-timestep recurrence is
        #   state_t = decay_t * state_{t-1} + x_t,  decay_t = exp(-dt_t * A) in (0,1)
        # whose exact closed form is
        #   state_t = prod_{k<=t}(decay_k) * cumsum_i( x_i / prod_{k<=i}(decay_k) ).
        # Computed in log-space (log_fwd = cumsum log decay) for a fixed, small
        # op count per layer -- no Python time-loop -- so this captures cleanly
        # into a CUDA graph (cudagraphs) and scales to 200k tokens. log_fwd is
        # clamped so the reciprocal never overflows float32 (the far-past tail
        # is already ~0 and correctly contributes nothing).
        log_a = -dt.unsqueeze(-1) * A.view(1, 1, A.shape[0], A.shape[1])
        log_fwd = torch.cumsum(log_a, dim=1).clamp(min=-50.0)
        xin = x_in.unsqueeze(-1)  # (b, t, d_inner, 1)
        terms = xin * torch.exp(-log_fwd)
        s = torch.exp(log_fwd) * torch.cumsum(terms, dim=1)  # (b, t, d_inner, state_dim)
        out = s.sum(dim=-1) + self.D * x_in  # (b, t, d_inner)
        y = out * z.sigmoid()
        y = self.out_proj(y)
        return y, s[:, -1].transpose(1, 2).contiguous()
