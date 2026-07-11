"""8-bit Adam optimizer (state quantized to int8, scale per tensor)."""
from __future__ import annotations

import torch

from torch.optim import Optimizer


class Adam8bit(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

    @staticmethod
    def _quantize(state, name, x: torch.Tensor):
        max_abs = x.abs().max().clamp_min(1e-8)
        q = torch.clamp(torch.round(x / max_abs * 127.0), -127, 127).to(torch.int8)
        state[name + "_q"] = q
        state[name + "_scale"] = max_abs

    @staticmethod
    def _dequantize(state, name):
        return state[name + "_q"].float() * state[name + "_scale"] / 127.0

    def step(self, closure=None):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                state["step"] += 1
                state["m"].mul_(beta1).add_(grad, alpha=1 - beta1)
                state["v"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                self._quantize(state, "m", state["m"])
                self._quantize(state, "v", state["v"])
                m_hat = state["m"] / (1 - beta1 ** state["step"])
                v_hat = state["v"] / (1 - beta2 ** state["step"])
                denom = v_hat.sqrt().add_(group["eps"])
                p.data.addcdiv_(m_hat, denom, value=-group["lr"])
        return None
