"""State-transition head: predicts latent delta, exit, and next action."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class StateTransitionConfig:
    dim: int = 16
    hidden: int = 32
    n_actions: int = 8


class StateTransitionHead(nn.Module):
    def __init__(self, cfg: StateTransitionConfig):
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.dim * 2, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.dim + 1 + cfg.n_actions),
        )

    def forward(self, state: torch.Tensor, context: torch.Tensor):
        x = torch.cat([state, context], dim=-1)
        out = self.net(x)
        dim = self.cfg.dim
        delta = out[:, :dim]
        exit_logit = out[:, dim]
        action_logits = out[:, dim + 1: dim + 1 + self.cfg.n_actions]
        return {"delta": delta, "exit_logit": exit_logit, "action_logits": action_logits}

    @torch.no_grad()
    def simulate(self, state: torch.Tensor, context: torch.Tensor):
        self.eval()
        out = self.forward(state, context)
        return {
            "exit_pred": (out["exit_logit"] > 0).long(),
            "action_pred": out["action_logits"].argmax(dim=-1),
            "delta": out["delta"],
        }
