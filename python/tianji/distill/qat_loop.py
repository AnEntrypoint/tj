"""Quantization-Aware Training loop: wraps the hybrid model with LoRA, runs
KD + replay, and tracks VRAM budget (4GB target)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from ..arch.hybrid import HybridStack, HybridConfig
from .lora import wrap_linear_with_lora, LoRAConfig, save_lora_state, load_lora_state
from .replay import ReplayBuffer
from .kd import kd_loss, make_stub_teacher, StubTeacher
from .router_alignment import RouterAlignment, apply_router_bias
from ..caps import ResourceBudget


@dataclass
class QATConfig:
    device: str = "cpu"
    lora_rank: int = 4
    vram_bytes: int = 4 * 1024 ** 3
    lr: float = 1e-3
    kd_alpha: float = 0.5


@dataclass
class QATStepResult:
    loss: float
    kd_loss: float
    vram_used_bytes: int
    aux_loss: float = 0.0


class _SteppingModel(nn.Module):
    def __init__(self, cfg: HybridConfig, vocab_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, cfg.dim)
        self.stack = HybridStack(cfg)
        self.head = wrap_linear_with_lora(nn.Linear(cfg.dim, vocab_size, bias=False), LoRAConfig(rank=4))

    def forward(self, ids):
        h = self.embed(ids)
        h, aux = self.stack(h)
        return self.head(h), aux


class QATLoop:
    def __init__(self, cfg: QATConfig, arch: HybridConfig, vocab_size: int = 64):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.vocab_size = vocab_size
        self.model = _SteppingModel(arch, vocab_size).to(self.device)
        self.teacher = make_stub_teacher(vocab_size).to(self.device)
        self.replay = ReplayBuffer(capacity=16)
        self.alignment = RouterAlignment.build(arch.dim)
        self.budget = ResourceBudget("vram", cfg.vram_bytes)
        self.opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=cfg.lr)
        # A stub teacher emits fixed random logits; distilling toward it only
        # injects noise, so KD is disabled unless a real teacher is supplied.
        self.kd_enabled = not isinstance(self.teacher, StubTeacher)
        self.kd_alpha = cfg.kd_alpha
        # The 4GB VRAM budget is a real invariant, not just a tracked number:
        # allocate the model's footprint up front and fail loudly if it exceeds.
        self._model_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        self.budget.allocate(self._model_bytes)
        self._closed = False

    def _vram(self) -> int:
        if self.device.type == "cuda":
            return torch.cuda.memory_allocated(self.device)
        # rough cpu accounting: params + buffers bytes
        total = 0
        for p in self.model.parameters():
            total += p.numel() * p.element_size()
        return total

    def step(self, input_ids: torch.Tensor, target_ids: torch.Tensor, source: str = "synthetic") -> QATStepResult:
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        self.opt.zero_grad(set_to_none=True)
        logits, aux = self.model(input_ids)
        ce = torch.nn.functional.cross_entropy(logits.view(-1, self.vocab_size), target_ids.view(-1))
        with torch.no_grad():
            teacher_logits = self.teacher(input_ids)
        kd = kd_loss(logits, teacher_logits.detach(), T=2.0)
        kd_term = (self.kd_alpha if self.kd_enabled else 0.0) * kd
        loss = ce + kd_term + 0.01 * aux
        loss.backward()
        self.opt.step()
        self.replay.push(input_ids.detach().cpu(), target_ids.detach().cpu())
        return QATStepResult(
            loss=float(loss.item()),
            kd_loss=float(kd.item()),
            vram_used_bytes=int(self._vram()),
            aux_loss=float(aux.item()),
        )

    @torch.no_grad()
    def hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return the hybrid stack's final hidden states (pre-head) for the
        given token ids -- used by the Engine as the agent's latent state."""
        input_ids = input_ids.to(self.device)
        h = self.model.embed(input_ids)
        h, _ = self.model.stack(h)
        return h

    def save_checkpoint(self, path: str) -> None:
        torch.save({"lora": save_lora_state(self.model), "cfg": self.cfg, "vocab_size": self.vocab_size}, path)

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        n = load_lora_state(self.model, ckpt["lora"])
        return n

    def close(self) -> None:
        self._closed = True
        self.budget.free(self._model_bytes)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
