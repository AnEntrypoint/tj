"""Elastic Weight Consolidation (EWC) for continual distillation."""
from __future__ import annotations

import torch

from .lora import wrap_linear_with_lora, LoRAConfig


class LoRAAdapter:
    def __init__(self, rank: int = 4):
        self.cfg = LoRAConfig(rank=rank)


class EWCState:
    def __init__(self, params, fisher):
        self.params = list(params)
        self.fisher = fisher


def compute_fisher(model: torch.nn.Module, dataloader, loss_fn) -> dict:
    fisher = {}
    for p in model.parameters():
        fisher[id(p)] = torch.zeros_like(p)
    model.eval()
    for x, y in dataloader:
        model.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                fisher[id(p)] += p.grad.detach() ** 2
    return fisher


def consolidate(model, fisher: dict, lam: float = 0.0) -> EWCState:
    return EWCState([p for p in model.parameters()], fisher)


def ewc_penalty(model: torch.nn.Module, ewc: EWCState, lam: float = 1.0) -> torch.Tensor:
    penalty = torch.zeros((), device="cpu", dtype=torch.float32)
    for p in model.parameters():
        f = ewc.fisher.get(id(p))
        if f is not None:
            penalty = penalty + (f * (p - p.detach()) ** 2).sum()
    return lam * penalty
