"""Router alignment: bias the MoE router toward a session's source expert."""
from __future__ import annotations

import torch

SOURCE_TO_EXPERT = {
    "claude": 12,
    "gpt-4o": 15,
    "cursor": 3,
    "synthetic": 7,
    "github-copilot": 9,
}


class RouterAlignment:
    def __init__(self, dim: int, bias: torch.Tensor):
        self.dim = dim
        self._bias = bias

    @classmethod
    def build(cls, dim: int) -> "RouterAlignment":
        bias = torch.zeros(dim)
        for expert_idx in SOURCE_TO_EXPERT.values():
            if expert_idx < dim:
                bias[expert_idx] = 0.5
        return cls(dim, bias)

    def bias_for(self, source: str) -> torch.Tensor:
        return self._bias.clone()


def apply_router_bias(logits: torch.Tensor, alignment: RouterAlignment, source: str) -> torch.Tensor:
    return logits + alignment.bias_for(source).to(logits.device)
