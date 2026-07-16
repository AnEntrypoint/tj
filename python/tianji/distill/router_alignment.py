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
    def __init__(self, n_experts: int, bias: torch.Tensor):
        self.n_experts = n_experts
        self._bias = bias

    @classmethod
    def build(cls, n_experts: int) -> "RouterAlignment":
        bias = torch.zeros(n_experts)
        for expert_idx in SOURCE_TO_EXPERT.values():
            if expert_idx < n_experts:
                bias[expert_idx] = 0.5
        return cls(n_experts, bias)

    def bias_for(self, source: str) -> torch.Tensor:
        bias = torch.zeros(self.n_experts)
        expert_idx = SOURCE_TO_EXPERT.get(source)
        if expert_idx is not None and expert_idx < self.n_experts:
            bias[expert_idx] = 0.5
        return bias


def apply_router_bias(logits: torch.Tensor, alignment: RouterAlignment, source: str) -> torch.Tensor:
    return logits + alignment.bias_for(source).to(logits.device)
