"""Replay buffer (ring buffer) for experience replay with DER++ support.

DER++ (Dark Experience Replay, NeurIPS 2020): stores logits alongside
inputs. During replay, applies KL divergence between current and stored
logits in addition to cross-entropy. This prevents catastrophic forgetting
better than pure replay — critical for continual learning across data sources.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import torch
import torch.nn.functional as F


class ReplayBuffer:
    def __init__(self, capacity: int = 64):
        self.capacity = capacity
        self._buf = deque(maxlen=capacity)

    def push(self, x: torch.Tensor, y: torch.Tensor,
             logits: Optional[torch.Tensor] = None) -> None:
        item = (x.detach().clone(), y.detach().clone())
        if logits is not None:
            item = item + (logits.detach().clone(),)
        self._buf.append(item)

    def sample(self, n: int):
        if len(self._buf) == 0:
            return None
        n = min(n, len(self._buf))
        idx = torch.randperm(len(self._buf))[:n].tolist()
        xs = [self._buf[i][0] for i in idx]
        ys = [self._buf[i][1] for i in idx]
        has_logits = len(self._buf[0]) > 2
        if has_logits:
            ls = [self._buf[i][2] for i in idx]
            return torch.stack(xs), torch.stack(ys), torch.stack(ls)
        return torch.stack(xs), torch.stack(ys)

    def derpp_loss(self, model, n: int, T: float = 2.0) -> torch.Tensor:
        """DER++ replay loss: CE + KL on stored logits."""
        batch = self.sample(n)
        if batch is None or len(batch) < 3:
            return torch.tensor(0.0)
        rx, ry, r_logits = batch
        out, _, _ = model(rx)
        ce = F.cross_entropy(out.view(-1, out.shape[-1]), ry.view(-1))
        s = F.log_softmax(out / T, dim=-1)
        t = F.softmax(r_logits / T, dim=-1)
        kl = F.kl_div(s, t, reduction="batchmean") * (T * T)
        return ce + 0.5 * kl

    def __len__(self) -> int:
        return len(self._buf)
