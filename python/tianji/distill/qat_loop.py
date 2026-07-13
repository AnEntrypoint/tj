"""Quantization-Aware Training loop: wraps the hybrid model with LoRA, runs
KD + replay, and tracks VRAM budget (4GB target)."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

# torch.compile (CUDA graphs / inductor) needs Triton. If it is missing we
# degrade gracefully to eager instead of crashing training at the first
# (lazily-compiled) forward.
try:
    import triton  # noqa: F401
    _TRITON_OK = True
except Exception:
    _TRITON_OK = False

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
    seq_len: int = 512


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
        # Cap this process at 4 GB (~67% of 6 GB) so a too-large allocation
        # errors cleanly (CUDA OOM) instead of taking down the display driver
        # with a hard OOM, which forces a GPU reset / restart. Training chunks
        # by --seq-len so a single step stays well under this budget.
        if self.device.type == "cuda":
            try:
                torch.cuda.set_per_process_memory_fraction(0.67)
            except Exception:
                pass
        self.model = _SteppingModel(arch, vocab_size).to(self.device)
        self.teacher = make_stub_teacher(vocab_size).to(self.device)
        self.replay = ReplayBuffer(capacity=16)
        self.alignment = RouterAlignment.build(arch.dim)
        self.budget = ResourceBudget("vram", cfg.vram_bytes)
        # capturable=True lets the optimizer step be captured inside a CUDA
        # graph (required for the cudagraph fast path on cuda).
        self.opt = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=cfg.lr, capturable=(self.device.type == "cuda"))
        # A stub teacher emits fixed random logits; distilling toward it only
        # injects noise, so KD is disabled unless a real teacher is supplied.
        self.kd_enabled = not isinstance(self.teacher, StubTeacher)
        self.kd_alpha = cfg.kd_alpha
        # The 4GB VRAM budget is a real invariant, not just a tracked number:
        # allocate the model's footprint up front and fail loudly if it exceeds.
        self._model_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        self.budget.allocate(self._model_bytes)
        self._closed = False
        # CUDA graphs cut the Python / kernel-launch overhead that dominates
        # this tiny dim-16 model, and (per directive) are used wherever possible
        # on cuda. On Triton-capable envs we use torch.compile(mode=reduce-
        # overhead); on Windows where Triton has no wheel we fall back to an
        # explicit torch.cuda.CUDAGraph capturing the fwd+bwd+optimizer step
        # (no Triton needed). An explicit TIANJI_COMPILE=0 disables both. Must
        # run after teacher / opt / kd_enabled exist (the captured region uses
        # them).
        self._graph = None
        _compile = os.environ.get("TIANJI_COMPILE", "1" if torch.cuda.is_available() else "0")
        if _compile != "0" and self.device.type == "cuda":
            if _TRITON_OK:
                try:
                    torch._dynamo.config.suppress_errors = True
                    self.model = torch.compile(self.model, mode="reduce-overhead", dynamic=True)
                except Exception as _e:  # pragma: no cover - environment dependent
                    print(f"[qat] torch.compile unavailable, using eager: {_e}", file=sys.stderr)
            # CUDA graph capture — the model must have no python-side control
            # flow or dynamic-shape ops inside the captured region.  The MoE
            # uses static-shape expert evaluation (every expert processes all
            # tokens) so there is no limit on layer count.
            else:
                self._setup_cudagraph()

    def _vram(self) -> int:
        if self.device.type == "cuda":
            return torch.cuda.memory_allocated(self.device)
        # rough cpu accounting: params + buffers bytes
        total = 0
        for p in self.model.parameters():
            total += p.numel() * p.element_size()
        return total

    def step(self, input_ids: torch.Tensor, target_ids: torch.Tensor,
             source: str = "synthetic", mask: torch.Tensor = None) -> QATStepResult:
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        if mask is not None:
            mask = mask.to(self.device).bool()
        # Fast path: replay the captured CUDA graph (fwd+bwd+optimizer step).
        if self._graph is not None:
            self._g_in.copy_(input_ids)
            self._g_tgt.copy_(target_ids)
            if mask is not None:
                self._g_mask.copy_(mask)
            self._graph.replay()
            loss = float(self._g_loss.item())
            self.replay.push(input_ids.detach().cpu(), target_ids.detach().cpu())
            return QATStepResult(
                loss=loss,
                kd_loss=0.0,
                vram_used_bytes=int(self._vram()),
                aux_loss=0.0,
            )
        self.opt.zero_grad(set_to_none=True)
        logits, aux = self.model(input_ids)
        ce = torch.nn.functional.cross_entropy(
            logits.view(-1, self.vocab_size), target_ids.view(-1), reduction="none")
        if mask is not None:
            m = mask.view(-1)
            ce = (ce * m).sum() / m.sum().clamp(min=1)
        else:
            ce = ce.mean()
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

    def _graph_forward_backward(self, inp: torch.Tensor, tgt: torch.Tensor,
                               mask: torch.Tensor) -> None:
        """The exact op sequence captured into the CUDA graph: forward, masked
        CE + (disabled) KD + aux loss, backward, optimizer step. ``_g_loss`` is
        written so the caller can read the loss back after replay."""
        logits, aux = self.model(inp)
        ce = torch.nn.functional.cross_entropy(
            logits.view(-1, self.vocab_size), tgt.view(-1), reduction="none")
        m = mask.view(-1)
        ce = (ce * m).sum() / m.sum().clamp(min=1)
        with torch.no_grad():
            teacher_logits = self.teacher(inp)
        kd = kd_loss(logits, teacher_logits.detach(), T=2.0)
        kd_term = (self.kd_alpha if self.kd_enabled else 0.0) * kd
        loss = ce + kd_term + 0.01 * aux
        self._g_loss.copy_(loss.detach())
        loss.backward()
        self.opt.step()

    def _setup_cudagraph(self) -> None:
        """Capture a reusable CUDA graph for the per-step fwd+bwd+optimizer.
        Explicit torch.cuda.CUDAGraph (no Triton needed, so it works on Windows).
        Any failure degrades to eager instead of crashing training."""
        try:
            L = int(self.cfg.seq_len)
            self._g_in = torch.zeros(1, L, dtype=torch.long, device=self.device)
            self._g_tgt = torch.zeros(1, L, dtype=torch.long, device=self.device)
            self._g_mask = torch.ones(1, L, dtype=torch.bool, device=self.device)
            self._g_loss = torch.zeros((), device=self.device)
            # Warmup so the captured region is traced with real shapes/ops.
            self._graph_forward_backward(self._g_in, self._g_tgt, self._g_mask)
            self.opt.zero_grad(set_to_none=True)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._graph_forward_backward(self._g_in, self._g_tgt, self._g_mask)
            self._graph = g
            print("[qat] CUDA graph captured (cudagraphs enabled)", file=sys.stderr)
        except Exception as _e:  # pragma: no cover - environment dependent
            self._graph = None
            # Best-effort recovery: a failed capture can leave the CUDA context
            # mid-capture, so synchronize before continuing on the eager path.
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            print(f"[qat] CUDA graph unavailable, using eager: {_e}", file=sys.stderr)

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
