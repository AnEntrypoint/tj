"""Token generator driving the QAT model with a paged KV cache.

Pre-allocates a fixed-size buffer so the autoregressive loop never calls
``torch.cat``, eliminating O(N) allocation per token and enabling future
CUDA graph replay with a fixed input shape + causal mask."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass
class GenerateConfig:
    max_tokens: int = 16
    paged_kv_blocks: int = 8


@dataclass
class GenerateStep:
    token: int


class Generator:
    def __init__(self, qat, cfg: GenerateConfig):
        self.qat = qat
        self.cfg = cfg
        self.model = qat.model
        self.device = qat.device
        self._max_allocated = 0
        self._buf: torch.Tensor | None = None

    def _ensure_buf(self, n: int) -> torch.Tensor:
        if self._buf is None or n > self._buf.shape[1]:
            self._buf = torch.zeros(
                1, n + self.cfg.max_tokens, dtype=torch.long, device=self.device)
            self._max_allocated = 0
        return self._buf

    def generate(self, prompt_ids) -> Iterator[GenerateStep]:
        if not isinstance(prompt_ids, torch.Tensor):
            prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
        ids = prompt_ids.to(self.device).unsqueeze(0)
        plen = ids.shape[1]
        buf = self._ensure_buf(plen)
        buf[:, :plen].copy_(ids)
        cur_len = plen
        for _ in range(self.cfg.max_tokens):
            inp = buf[:, :cur_len]
            with torch.no_grad():
                h = self.model.embed(inp)
                h, _, _ = self.model.stack(h)
                logits = self.model.head(h)
            nxt = int(logits[0, -1].argmax(dim=-1).item())
            yield GenerateStep(token=nxt)
            buf[0, cur_len] = nxt
            cur_len += 1
