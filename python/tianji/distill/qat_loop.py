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
from .kd import kd_loss, make_stub_teacher, StubTeacher, reverse_kd_loss, jsd_loss, topk_kd_loss
from .router_alignment import RouterAlignment
from .ewc import EWCState
from ..caps import ResourceBudget


@dataclass
class QATConfig:
    device: str = "cpu"
    lora_rank: int = 4
    vram_bytes: int = 4 * 1024 ** 3
    lr: float = 1e-3
    kd_alpha: float = 0.5
    seq_len: int = 512
    precision: str = "fp16"     # "fp32", "fp16", "bf16"
    warmup_steps: int = 100     # linear LR warmup
    grad_clip: float = 1.0      # gradient clipping norm (0 = disabled)
    lr_decay: float = 0.1       # cosine decay to this fraction of initial LR
    kd_mode: str = "jsd"        # "forward", "reverse", "jsd", "topk"


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
        self._step_count = 0
        self._lr_scheduler = None
        self._lr_warmed_up = False
        if cfg.warmup_steps > 0 or cfg.lr_decay < 1.0:
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
            warmup = LinearLR(self.opt, start_factor=0.01, end_factor=1.0,
                              total_iters=cfg.warmup_steps)
            cosine = CosineAnnealingLR(self.opt, T_max=10000, eta_min=cfg.lr * cfg.lr_decay)
            self._lr_scheduler = SequentialLR(self.opt, schedulers=[warmup, cosine],
                                               milestones=[cfg.warmup_steps])
        self.kd_alpha = cfg.kd_alpha
        self._model_bytes = sum(
            p.numel() * p.element_size() for p in self.model.parameters())
        self.budget.allocate(self._model_bytes)
        self._closed = False

        # EWC for continual learning across data sources.
        self._ewc: Optional["EWCState"] = None
        self._ewc_lambda = 0.0  # disabled by default

        # Automatic mixed precision (AMP) — saves ~40% VRAM on CUDA.
        self._amp_dtype = None
        self._scaler = None
        if self.device.type == "cuda" and cfg.precision != "fp32":
            if cfg.precision == "bf16" and torch.cuda.is_bf16_supported():
                self._amp_dtype = torch.bfloat16
            elif cfg.precision == "fp16":
                self._amp_dtype = torch.float16
                self._scaler = torch.amp.GradScaler("cuda")
            if self._amp_dtype is not None:
                print(
                    f"[qat] AMP enabled: dtype={self._amp_dtype} scaler={'yes' if self._scaler else 'no'}",
                    file=sys.stderr, flush=True)

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
        _no_graph = os.environ.get("TIANJI_CUDAGRAPH", "1") == "0"
        if _compile != "0" and self.device.type == "cuda":
            # CUDA graphs are the default acceleration path — they work on
            # both Windows and Linux, are more memory-predictable than
            # torch.compile, and capture the full fwd+bwd in one replayable
            # graph.  Only skip if explicitly disabled (TIANJI_CUDAGRAPH=0)
            # or if the model is too large for graph capture.
            if not _no_graph:
                self._setup_cudagraph()
            # If cudagraph capture failed or was skipped, try torch.compile
            # on Linux as a secondary acceleration path (Triton-dependent).
            if self._graph is None and _TRITON_OK:
                try:
                    torch._dynamo.config.suppress_errors = True
                    self.model = torch.compile(
                        self.model, mode="reduce-overhead", dynamic=True)
                except Exception as _e:
                    print(
                        f"[qat] torch.compile unavailable, using eager: {_e}",
                        file=sys.stderr, flush=True)
        if self._graph is None and self.device.type == "cuda":
            # If neither graph capture nor compile succeeded, we can still
            # train — just slower.  Log the fallback once.
            if _compile != "0":
                print(
                    "[qat] no graph acceleration available, using eager mode",
                    file=sys.stderr, flush=True)

    def _kd_loss(self, student_logits, teacher_logits):
        """Compute KD loss using the configured mode."""
        if self.cfg.kd_mode == "reverse":
            return reverse_kd_loss(student_logits, teacher_logits, T=2.0)
        elif self.cfg.kd_mode == "jsd":
            return jsd_loss(student_logits, teacher_logits, T=2.0)
        elif self.cfg.kd_mode == "topk":
            return topk_kd_loss(student_logits, teacher_logits, T=2.0, top_k=40)
        return kd_loss(student_logits, teacher_logits, T=2.0)

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
            if self._scaler is not None:
                self._scaler.step(self.opt)
                self._scaler.update()
            else:
                self.opt.step()
            self._step_count += 1
            if self._lr_scheduler is not None and self._lr_warmed_up:
                self._lr_scheduler.step()
            self._lr_warmed_up = True
            if self.cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.cfg.grad_clip)
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
        with torch.amp.autocast("cuda", enabled=self._amp_dtype is not None, dtype=self._amp_dtype):
            logits, aux, next_states = self.model(input_ids, state_tuple)
        ce = torch.nn.functional.cross_entropy(
            logits.view(-1, self.vocab_size),
            target_ids.view(-1), reduction="none")
        if mask is not None:
            m = mask.view(-1)
            ce = (ce * m).sum() / m.sum().clamp(min=1)
        else:
            ce = ce.mean()
        with torch.amp.autocast("cuda", enabled=self._amp_dtype is not None, dtype=self._amp_dtype):
            with torch.no_grad():
                teacher_logits = self.teacher(input_ids)
        kd = self._kd_loss(logits, teacher_logits.detach())
        kd_term = (self.kd_alpha if self.kd_enabled else 0.0) * kd
        loss = ce + kd_term + 0.01 * aux
        if self._ewc is not None and self._ewc_lambda > 0:
            loss = loss + self._ewc.penalty(self.model, self._ewc_lambda)
        if self._scaler is not None:
            self._scaler.scale(loss).backward()
            self._scaler.step(self.opt)
            self._scaler.update()
        else:
            loss.backward()
            self.opt.step()
        if self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.cfg.grad_clip)
        self._step_count += 1
        if self._lr_scheduler is not None:
            self._lr_scheduler.step()
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

        AMP autocast is disabled during graph capture (the capture itself
        triggers CUDAGeneratorImpl::current_seed errors with autocast on).
        The graph runs in fp32 which is fine for the cudagraph path — the
        speedup from graph replay outweighs the precision loss.
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
        kd = self._kd_loss(logits, teacher_logits.detach())
        self._g_kd.copy_(kd.detach())
        self._g_aux.copy_(aux.detach())
        kd_term = (self.kd_alpha if self.kd_enabled else 0.0) * kd
        loss = ce + kd_term + 0.01 * aux
        self._g_loss.copy_(loss.detach())
        if self._ewc is not None and self._ewc_lambda > 0:
            loss = loss + self._ewc.penalty(self.model, self._ewc_lambda)
        if self._scaler is not None:
            self._scaler.scale(loss).backward()
        else:
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

            # Set deterministic seeds before warmup + capture to avoid
            # CUDAGeneratorImpl::current_seed errors during graph capture.
            _saved_cpu_rng = torch.get_rng_state()
            _saved_cuda_rng = torch.cuda.get_rng_state()
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)

            # Warmup on the default stream to create p.grad tensors.
            for _ in range(3):
                self._graph_forward_backward(
                    self._g_in, self._g_tgt, self._g_mask)
            torch.cuda.synchronize()

            self.opt.zero_grad(set_to_none=False)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._graph_forward_backward(
                    self._g_in, self._g_tgt, self._g_mask)
            self._graph = g

            # Restore original RNG state.
            torch.set_rng_state(_saved_cpu_rng)
            torch.cuda.set_rng_state(_saved_cuda_rng)

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

    def set_ewc(self, ewc: EWCState, lam: float = 1.0) -> None:
        """Enable EWC continual learning with the given state and lambda."""
        self._ewc = ewc
        self._ewc_lambda = lam

    def clear_ewc(self) -> None:
        """Disable EWC penalty."""
        self._ewc = None
        self._ewc_lambda = 0.0

    def close(self) -> None:
        self._closed = True
        self.budget.free(self._model_bytes)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
