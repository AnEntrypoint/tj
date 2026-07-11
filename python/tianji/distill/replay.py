"""Replay buffer (ring buffer) for experience replay."""
from __future__ import annotations

from collections import deque

import torch


class ReplayBuffer:
    def __init__(self, capacity: int = 64):
        self.capacity = capacity
        self._buf = deque(maxlen=capacity)

    def push(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self._buf.append((x.detach().clone(), y.detach().clone()))

    def sample(self, n: int):
        if len(self._buf) == 0:
            return None
        n = min(n, len(self._buf))
        idx = torch.randperm(len(self._buf))[:n].tolist()
        xs = [self._buf[i][0] for i in idx]
        ys = [self._buf[i][1] for i in idx]
        return torch.stack(xs), torch.stack(ys)

    def __len__(self) -> int:
        return len(self._buf)
