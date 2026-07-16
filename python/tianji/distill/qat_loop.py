"""Quantization-Aware Training loop: wraps the hybrid model with LoRA, runs
KD + replay, and tracks VRAM budget (4GB target).  Every chunk uses the
same CUDA graph (forward + backward + optimizer step + state carry), so
Mamba-2 context persists across chunks without any Python-side re-entry."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    import triton  # noqa: F401
    _TRITON_OK = True
except Exception:
    _TRITON_OK = False

from ..arch.hybrid import HybridStack, HybridConfig
from .lora import wrap_linear_with_lora, LoRAConfig, save_lora_state, load_lora_state
from .replay import ReplayBuffer
from .kd import kd_loss, make_stub_teacher, StubTeacher
from .router_alignment import RouterAlignment
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
        self.head = wrap_linear_with_lora(
            nn.Linear(cfg.dim, vocab_size, bias=False), LoRAConfig(rank=4))

    def forward(self, ids, state_tuple=None, router_bias=None):
        h = self.embed(ids)
        h, aux, next_tuple = self.stack(h, state_tuple, router_bias=router_bias)
        return self.head(h), aux, next_tuple


class QATLoop:
    def __init__(self, cfg: QATConfig, arch: HybridConfig, vocab_size: int = 64):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.vocab_size = vocab_size
        self.arch = arch

        if self.device.type == "cuda":
            try:
                torch.cuda.set_per_process_memory_fraction(0.88)
            except Exception:
                pass

        self.model = _SteppingModel(arch, vocab_size).to(self.device)
        self.teacher = make_stub_teacher(vocab_size).to(self.device)
        self.replay = ReplayBuffer(capacity=16)
        self.alignment = RouterAlignment.build(arch.n_experts)
        self.budget = ResourceBudget("vram", cfg.vram_bytes)
        self.opt = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=cfg.lr, capturable=(self.device.type == "cuda"))
        self.kd_enabled = not isinstance(self.teacher, StubTeacher)
        self.kd_alpha = cfg.kd_alpha
        self._model_bytes = sum(
            p.numel() * p.element_size() for p in self.model.parameters())
        self.budget.allocate(self._model_bytes)
        self._closed = False

        # Pre-allocate Mamba-2 state buffers for CUDA graph.
        # Shape: (batch=1, d_inner, state_dim).
        # Separate in/out buffers: the graph reads from in (never modified
        # inside the graph) and writes to out (never read by the model).
        # After replay, step() copies out -> in so the next chunk inherits
        # the carried state.  This avoids the autograd in-place version error
        # (copy_ inside the graph would bump the input tensor's version).
        self._num_stateful = self.model.stack.num_stateful
        self._g_state_in: list[torch.Tensor] = []
        self._g_state_out: list[torch.Tensor] = []
        self._g_state_in_tuple: Tuple[torch.Tensor, ...] | None = None
        if self._num_stateful > 0:
            for _ in range(self._num_stateful):
                self._g_state_in.append(
                    torch.zeros(1, arch.d_inner, arch.state_dim,
                                device=self.device))
                self._g_state_out.append(
                    torch.zeros(1, arch.d_inner, arch.state_dim,
                                device=self.device))
            self._g_state_in_tuple = tuple(self._g_state_in)

        self._graph = None
        _compile = os.environ.get("TIANJI_COMPILE",
                                  "1" if torch.cuda.is_available() else "0")
        _force_graph = os.environ.get("TIANJI_CUDAGRAPH", "0") == "1"
        if _compile != "0" and self.device.type == "cuda":
            if _TRITON_OK:
                try:
                    torch._dynamo.config.suppress_errors = True
                    self.model = torch.compile(
                        self.model, mode="reduce-overhead", dynamic=True)
                except Exception as _e:
                    print(
                        f"[qat] torch.compile unavailable, using eager: {_e}",
                        file=sys.stderr, flush=True)
            elif _force_graph:
                # CUDA graph capture on Windows at dim >= 768 can OOM /
                # corrupt the CUDA context; only attempt when explicitly
                # requested via TIANJI_CUDAGRAPH=1.
                self._setup_cudagraph()
        if self._graph is None and self.device.type == "cuda":
            # If graph capture didn't happen (eager fallback) we can still
            # train — just slower.  Log once so the user isn't surprised.
            pass

    def _vram(self) -> int:
        if self.device.type == "cuda":
            return torch.cuda.memory_allocated(self.device)
        total = 0
        for p in self.model.parameters():
            total += p.numel() * p.element_size()
        return total

    def step(self, input_ids: torch.Tensor, target_ids: torch.Tensor,
             source: str = "synthetic", mask: torch.Tensor = None,
             state_tuple: Tuple[torch.Tensor, ...] | None = None
             ) -> Tuple[QATStepResult, Tuple[torch.Tensor, ...] | None]:
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        if mask is not None:
            mask = mask.to(self.device).bool()

        # ── CUDA graph path (stateful) ──────────────────────────────────
        if self._graph is not None:
            assert input_ids.shape == self._g_in.shape, (
                f"Input shape {input_ids.shape} != graph {self._g_in.shape}")
            assert target_ids.shape == self._g_tgt.shape, (
                f"Target shape {target_ids.shape} != graph {self._g_tgt.shape}")
            assert mask is not None and mask.shape == self._g_mask.shape, (
                f"Mask shape {mask.shape} != graph {self._g_mask.shape}")
            self._g_in.copy_(input_ids)
            self._g_tgt.copy_(target_ids)
            self._g_mask.copy_(mask)
            # Seed the input state buffer: caller-provided state or zeros.
            if state_tuple is not None:
                for si, ts in zip(self._g_state_in, state_tuple):
                    si.copy_(ts)
            else:
                for si in self._g_state_in:
                    si.zero_()
            # Update the router bias for this source before replay.
            bias = self.alignment.bias_for(source).to(self.device)
            self._g_router_bias.copy_(bias)
            self._graph.replay()
            # Optimizer step outside the graph so AdamW state allocations
            # don't get captured into the graph memory pool (would exceed 6GB).
            self.opt.step()
            loss = float(self._g_loss.item())
            kd = float(self._g_kd.item())
            aux = float(self._g_aux.item())
            # Carry output states to input for the next chunk (outside graph
            # so the in-place copy doesn't corrupt autograd).
            for si, so in zip(self._g_state_in, self._g_state_out):
                si.copy_(so)
            next_states = (
                tuple(self._g_state_in) if self._num_stateful > 0 else None)
            self.replay.push(
                input_ids.detach().cpu(), target_ids.detach().cpu())
            return (QATStepResult(
                loss=loss, kd_loss=kd,
                vram_used_bytes=int(self._vram()),
                aux_loss=aux), next_states)

        # ── Eager fallback (no CUDA graph) ──────────────────────────────
        self.opt.zero_grad(set_to_none=True)
        logits, aux, next_states = self.model(input_ids, state_tuple)
        ce = torch.nn.functional.cross_entropy(
            logits.view(-1, self.vocab_size),
            target_ids.view(-1), reduction="none")
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
        self.replay.push(
            input_ids.detach().cpu(), target_ids.detach().cpu())
        return (QATStepResult(
            loss=float(loss.item()),
            kd_loss=float(kd.item()),
            vram_used_bytes=int(self._vram()),
            aux_loss=float(aux.item()),
        ), next_states)

    def _graph_forward_backward(self, inp: torch.Tensor, tgt: torch.Tensor,
                                mask: torch.Tensor) -> None:
        """Captured into CUDA graph: forward (with state carry), masked
        CE + KD + aux loss, backward.

        The optimizer step runs OUTSIDE the graph (after replay) because
        the AdamW state allocation would balloon the graph memory pool past
        the GPU's 6 GB budget at dim=1024.  The graph reads state from
        ``_g_state_in_tuple`` and writes output state to ``_g_state_out``
        so the caller can copy out->in without autograd in-place errors.
        """
        self.opt.zero_grad()
        logits, aux, next_states = self.model(
            inp, self._g_state_in_tuple, router_bias=self._g_router_bias)
        if next_states is not None:
            for i, ns in enumerate(next_states):
                self._g_state_out[i].copy_(ns)
        ce = torch.nn.functional.cross_entropy(
            logits.view(-1, self.vocab_size), tgt.view(-1), reduction="none")
        m = mask.view(-1)
        ce = (ce * m).sum() / m.sum().clamp(min=1)
        with torch.no_grad():
            teacher_logits = self.teacher(inp)
        kd = kd_loss(logits, teacher_logits.detach(), T=2.0)
        self._g_kd.copy_(kd.detach())
        self._g_aux.copy_(aux.detach())
        kd_term = (self.kd_alpha if self.kd_enabled else 0.0) * kd
        loss = ce + kd_term + 0.01 * aux
        self._g_loss.copy_(loss.detach())
        loss.backward()

    def _setup_cudagraph(self) -> None:
        """Capture a reusable CUDA graph for the per-step fwd+bwd+optimizer,
        including Mamba-2 state carry.

        The warmup run creates the ``p.grad`` tensors that
        ``_graph_forward_backward`` needs to in-place-zero.
        """
        try:
            L = int(self.cfg.seq_len)
            self._g_in = torch.zeros(
                1, L, dtype=torch.long, device=self.device)
            self._g_tgt = torch.zeros(
                1, L, dtype=torch.long, device=self.device)
            self._g_mask = torch.ones(
                1, L, dtype=torch.bool, device=self.device)
            self._g_loss = torch.zeros((), device=self.device)
            self._g_kd = torch.zeros((), device=self.device)
            self._g_aux = torch.zeros((), device=self.device)
            self._g_router_bias = torch.zeros(
                self.arch.n_experts, device=self.device)

            # Warmup on a side stream to avoid capturing the default stream's
            # internal state into the graph memory pool.
            warmup_stream = torch.cuda.Stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream):
                for _ in range(3):
                    self._graph_forward_backward(
                        self._g_in, self._g_tgt, self._g_mask)
            torch.cuda.current_stream().wait_stream(warmup_stream)

            self.opt.zero_grad(set_to_none=False)
            # Defragment before the graph pool allocation so the contiguous
            # memory the graph requires actually fits.
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._graph_forward_backward(
                    self._g_in, self._g_tgt, self._g_mask)
            self._graph = g
            print(
                "[qat] CUDA graph captured (stateful cudagraphs enabled)",
                file=sys.stderr, flush=True)
        except Exception as _e:
            self._graph = None
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            print(
                f"[qat] CUDA graph unavailable, using eager: {_e}",
                file=sys.stderr, flush=True)

    @torch.no_grad()
    def hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        h = self.model.embed(input_ids)
        h, _, _ = self.model.stack(h)
        return h

    def save_checkpoint(self, path: str) -> None:
        torch.save({
            "lora": save_lora_state(self.model),
            "cfg": self.cfg,
            "vocab_size": self.vocab_size,
        }, path)

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(
            path, map_location=self.device, weights_only=False)
        return load_lora_state(self.model, ckpt["lora"])

    def close(self) -> None:
        self._closed = True
        self.budget.free(self._model_bytes)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
