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
    """Forward KL divergence: KL(teacher || student)."""
    s = F.log_softmax(student_logits / T, dim=-1)
    t = F.softmax(teacher_logits / T, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (T * T)


def reverse_kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float = 2.0) -> torch.Tensor:
    """Reverse KL divergence: KL(student || teacher). Mode-seeking behavior.
    Better for small student models (MiniLLM, ICLR 2024)."""
    s = F.softmax(student_logits / T, dim=-1)
    t = F.log_softmax(teacher_logits / T, dim=-1)
    return F.kl_div(t, s, reduction="batchmean") * (T * T)


def jsd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float = 2.0) -> torch.Tensor:
    """Generalized Jensen-Shannon divergence: balanced between forward and reverse KL.
    More stable than pure reverse KL (distillanything, 2026)."""
    s = F.softmax(student_logits / T, dim=-1)
    t = F.softmax(teacher_logits / T, dim=-1)
    m = 0.5 * (s + t)
    kl_sm = F.kl_div(F.log_softmax(student_logits / T, dim=-1), m, reduction="batchmean")
    kl_tm = F.kl_div(F.log_softmax(teacher_logits / T, dim=-1), m, reduction="batchmean")
    return 0.5 * (kl_sm + kl_tm) * (T * T)


def topk_kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                 T: float = 2.0, top_k: int = 0) -> torch.Tensor:
    """KD loss with top-k truncation. Only distills the top-k teacher logits,
    reducing noise from low-probability tokens. Essential for small vocabularies."""
    if top_k <= 0:
        return kd_loss(student_logits, teacher_logits, T)
    s = F.log_softmax(student_logits / T, dim=-1)
    t = F.softmax(teacher_logits / T, dim=-1)
    topk_vals, topk_idx = torch.topk(t, top_k, dim=-1)
    mask = torch.zeros_like(t).scatter_(-1, topk_idx, 1.0)
    t_masked = t * mask
    t_masked = t_masked / t_masked.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return F.kl_div(s, t_masked, reduction="batchmean") * (T * T)


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
