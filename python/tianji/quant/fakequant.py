"""Fake integer quantization (straight-through) for QAT."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _fake_quant(x: torch.Tensor, bits: int):
    qmax = 2 ** bits - 1
    scale = (x.max() - x.min()).clamp_min(1e-8) / qmax
    zero = torch.round(-x.min() / scale)
    q = torch.clamp(torch.round(x / scale + zero), 0, qmax)
    return (q - zero) * scale


def fakequant_int4(x: torch.Tensor) -> torch.Tensor:
    return x + (_fake_quant(x, 4) - x).detach()


class FakeQuantLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = fakequant_int4(self.weight)
        return F.linear(x, w, self.bias)


def apply_fakequant_to_linear(module: nn.Module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, FakeQuantLinear(child.in_features, child.out_features, child.bias is not None))
        else:
            apply_fakequant_to_linear(child)
    return module


def materialize_real_int4(module: nn.Module):
    for name, child in list(module.named_children()):
        if isinstance(child, FakeQuantLinear):
            quant = _fake_quant(child.weight, 4)
            real = nn.Linear(child.in_features, child.out_features, child.bias is not None)
            with torch.no_grad():
                real.weight.copy_(quant)
                if child.bias is not None:
                    real.bias.copy_(child.bias)
            setattr(module, name, real)
        else:
            materialize_real_int4(child)
    return module
