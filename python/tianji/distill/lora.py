"""LoRA adapters for linear layers with merge/save/load."""
from __future__ import annotations

import torch
import torch.nn as nn


class LoRAConfig:
    def __init__(self, rank: int = 4, alpha: float = 1.0):
        self.rank = rank
        self.alpha = alpha


class LoRAAdapter:
    def __init__(self, rank: int = 4):
        self.cfg = LoRAConfig(rank=rank)


class _LoRA(nn.Module):
    def __init__(self, rank: int, in_f: int, out_f: int, alpha: float):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.zeros(rank, in_f))
        self.B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)


class _LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, cfg: LoRAConfig):
        super().__init__()
        self.base = base
        in_f, out_f = base.in_features, base.out_features
        self.lora = _LoRA(cfg.rank, in_f, out_f, cfg.alpha)

    @property
    def A(self):
        return self.lora.A

    @property
    def B(self):
        return self.lora.B

    @property
    def scaling(self):
        return self.lora.scaling

    def forward(self, x):
        return self.base(x) + (x @ self.lora.A.T @ self.lora.B.T) * self.lora.scaling


def wrap_linear_with_lora(lin: nn.Linear, cfg: LoRAConfig) -> nn.Module:
    return _LoRALinear(lin, cfg)


def wrap_model_with_lora(model: nn.Module, cfg: LoRAConfig, names=("linear",)):
    for name, child in model.named_modules():
        if isinstance(child, nn.Linear):
            setattr(model, name.split(".")[-1], _LoRALinear(child, cfg))
    return model


def merge_lora(wrapped: nn.Module) -> int:
    n = 0
    if isinstance(wrapped, _LoRALinear):
        with torch.no_grad():
            wrapped.base.weight.add_(wrapped.lora.B @ wrapped.lora.A * wrapped.lora.scaling)
            wrapped.lora.A.zero_()
            wrapped.lora.B.zero_()
        n += 1
    return n


def save_lora_state(wrapped: nn.Module) -> dict:
    state = {}
    n = 0
    for module in wrapped.modules():
        if isinstance(module, _LoRALinear):
            key = f"lora_{n}"
            state[key] = {"A": module.A.detach(), "B": module.B.detach(), "rank": module.lora.rank, "scaling": module.lora.scaling}
            n += 1
    state["_count"] = n
    return state


def load_lora_state(wrapped: nn.Module, state: dict) -> int:
    n = state.get("_count", 0)
    modules = [m for m in wrapped.modules() if isinstance(m, _LoRALinear)]
    count = min(n, len(modules))
    for i in range(count):
        key = f"lora_{i}"
        if key not in state:
            continue
        with torch.no_grad():
            modules[i].A.copy_(state[key]["A"])
            modules[i].B.copy_(state[key]["B"])
    return count
