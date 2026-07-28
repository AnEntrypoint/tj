"""Elastic Weight Consolidation (EWC) for continual distillation.

Prevents catastrophic forgetting when training sequentially on different
data sources (e.g. switching between ccsniff sessions and public HF datasets).
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional

import torch
import torch.nn as nn


class EWCState:
    """Captured EWC importance weights and reference parameters.

    Stores the Fisher information matrix diagonal and the parameter values
    at the time of consolidation, keyed by parameter name (not id(), which
    changes after optimizer steps).
    """

    def __init__(self, fisher: Dict[str, torch.Tensor], ref_params: Dict[str, torch.Tensor]):
        self.fisher = fisher
        self.ref_params = ref_params

    def penalty(self, model: nn.Module, lam: float = 1.0) -> torch.Tensor:
        """Compute EWC penalty: sum_i F_i * (theta_i - theta_ref_i)^2."""
        total = torch.tensor(0.0)
        for name, p in model.named_parameters():
            f = self.fisher.get(name)
            ref = self.ref_params.get(name)
            if f is not None and ref is not None:
                f = f.to(p.device)
                ref = ref.to(p.device)
                total = total + (f * (p - ref) ** 2).sum()
        return lam * total


def compute_fisher(
    model: nn.Module,
    loss_fn,
    steps: int = 10,
    device: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """Approximate Fisher information diagonal from model gradients.

    Uses a few forward/backward passes with the current model state.
    For production use, accumulate over a representative sample of the
    training data.

    Args:
        model: The model to compute Fisher for.
        loss_fn: A callable that returns a scalar loss (no args needed).
        steps: Number of Monte Carlo samples for Fisher estimation.
        device: Device to run on.

    Returns:
        Dict mapping parameter names to Fisher diagonal tensors.
    """
    fisher: Dict[str, torch.Tensor] = {}
    model.train()
    for _ in range(steps):
        model.zero_grad(set_to_none=True)
        try:
            loss = loss_fn()
            if not isinstance(loss, torch.Tensor):
                continue
            loss.backward()
        except Exception:
            continue
        for name, p in model.named_parameters():
            if p.grad is not None:
                grad_sq = p.grad.detach() ** 2
                if name in fisher:
                    fisher[name] = fisher[name] + grad_sq
                else:
                    fisher[name] = grad_sq.clone()
    # Average over steps
    for name in fisher:
        fisher[name] = fisher[name] / max(1, steps)
    return fisher


def consolidate(model: nn.Module, fisher: Dict[str, torch.Tensor]) -> EWCState:
    """Create an EWC state from current model parameters and Fisher info.

    Captures both the Fisher diagonal and the current parameter values
    (as reference for the penalty term).
    """
    ref_params = {}
    for name, p in model.named_parameters():
        ref_params[name] = p.detach().cpu().clone()
    # Move fisher to CPU for storage
    fisher_cpu = {name: f.detach().cpu().clone() for name, f in fisher.items()}
    return EWCState(fisher_cpu, ref_params)


def ewc_penalty(model: nn.Module, ewc: EWCState, lam: float = 1.0) -> torch.Tensor:
    """Convenience: compute EWC penalty from a stored EWCState."""
    return ewc.penalty(model, lam)