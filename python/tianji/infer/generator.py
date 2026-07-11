"""Token generator driving the QAT model with a paged KV cache."""
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

    def generate(self, prompt_ids) -> Iterator[GenerateStep]:
        if not isinstance(prompt_ids, torch.Tensor):
            prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
        ids = prompt_ids.to(self.device).unsqueeze(0)
        for _ in range(self.cfg.max_tokens):
            with torch.no_grad():
                h = self.model.embed(ids)
                h, _ = self.model.stack(h)
                logits = self.model.head(h)
            nxt = int(logits[0, -1].argmax(dim=-1).item())
            yield GenerateStep(token=nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=self.device, dtype=torch.long)], dim=1)
