"""int2 KV-cache quantization with packing into int8 containers."""
from __future__ import annotations

import torch

INT2_MIN = -2
INT2_MAX = 1


def quant_int2(x: torch.Tensor):
    qmax = INT2_MAX
    qmin = INT2_MIN
    scale = (x.abs().max() / 2.0).clamp_min(1e-8)
    q = torch.clamp(torch.round(x / scale), qmin, qmax)
    return q.to(torch.int8), scale


def dequant_int2(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale


def pack_int2(q: torch.Tensor) -> torch.Tensor:
    q = q.to(torch.int32).clamp(INT2_MIN, INT2_MAX) - INT2_MIN
    flat = q.reshape(-1)
    pad = (-flat.numel()) % 4
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=torch.int32)])
    flat = flat.reshape(-1, 4)
    packed = flat[:, 0] | (flat[:, 1] << 2) | (flat[:, 2] << 4) | (flat[:, 3] << 6)
    return packed.to(torch.int8)


def unpack_int2(packed: torch.Tensor, n: int) -> torch.Tensor:
    packed = packed.to(torch.int32)
    q0 = packed & 3
    q1 = (packed >> 2) & 3
    q2 = (packed >> 4) & 3
    q3 = (packed >> 6) & 3
    q = torch.stack([q0, q1, q2, q3], dim=1).reshape(-1)
    q = q[:n].to(torch.int8)
    return q + INT2_MIN
