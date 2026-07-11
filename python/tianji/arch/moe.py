"""Mixture-of-Experts layer with shared experts and auxiliary load-balancing loss."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MoEConfig:
    dim: int = 16
    n_experts: int = 4
    n_active: int = 2
    shared_experts: int = 1
    expert_hidden: int = 32


class _Expert(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc = nn.Linear(dim, hidden)
        self.gate = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.gate(torch.nn.functional.relu(self.fc(x)))


class MoELayer(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        self.router = nn.Linear(cfg.dim, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList([_Expert(cfg.dim, cfg.expert_hidden) for _ in range(cfg.n_experts)])
        self.shared = nn.ModuleList([_Expert(cfg.dim, cfg.expert_hidden) for _ in range(cfg.shared_experts)])

    def forward(self, x: torch.Tensor):
        b, t, d = x.shape
        flat = x.reshape(-1, d)
        logits = self.router(flat)
        probs = torch.softmax(logits, dim=-1)
        topk = min(self.cfg.n_active, self.cfg.n_experts)
        w, idx = probs.topk(topk, dim=-1)
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)
        out = torch.zeros_like(flat)
        for k in range(topk):
            expert_id = idx[:, k]
            weight = w[:, k].unsqueeze(-1)
            for e in range(self.cfg.n_experts):
                mask = (expert_id == e)
                if mask.any():
                    out[mask] += weight[mask] * self.experts[e](flat[mask])
        for s in self.shared:
            out += s(flat)
        aux = self._aux(probs, idx, topk)
        return out.reshape(b, t, d), aux

    def _aux(self, probs, idx, topk):
        f = torch.zeros(self.cfg.n_experts, device=probs.device, dtype=probs.dtype)
        for k in range(topk):
            f.scatter_add_(0, idx[:, k], torch.ones_like(idx[:, k], dtype=probs.dtype))
        f = f / (f.sum() + 1e-8)
        mean_p = probs.mean(dim=0)
        return (f * mean_p).sum() * self.cfg.n_experts
