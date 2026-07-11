"""Knowledge distillation loss and a deterministic stub teacher."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KDConfig:
    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        self.temperature = temperature
        self.alpha = alpha


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float = 2.0) -> torch.Tensor:
    s = F.log_softmax(student_logits / T, dim=-1)
    t = F.softmax(teacher_logits / T, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (T * T)


class StubTeacher(nn.Module):
    def __init__(self, vocab_size: int = 32, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("w", torch.randn(vocab_size, vocab_size, generator=g))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids.long(), self.w)


class Teacher:
    def __init__(self, model: nn.Module):
        self.model = model

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        return self.model(ids)


def make_stub_teacher(vocab_size: int = 32, seed: int = 0) -> StubTeacher:
    return StubTeacher(vocab_size=vocab_size, seed=seed)
