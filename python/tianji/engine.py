"""Tianji training engine: ties vocab + hybrid model + QAT loop + state head."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .tokens.apt import Vocab, encode, embed
from .arch.hybrid import HybridConfig
from .distill.qat_loop import QATLoop, QATConfig, QATStepResult
from .state.transition import StateTransitionHead, StateTransitionConfig
from .protocol import Frame, Trajectory


@dataclass
class EngineConfig:
    device: str = "cpu"
    seq_len: int = 8
    batch_size: int = 1


@dataclass
class StepResult:
    qat: QATStepResult
    state_loss: float


@dataclass
class SimulateResult:
    delta: torch.Tensor
    exit_pred: torch.Tensor
    action_pred: torch.Tensor


class Engine:
    def __init__(self, vocab: Vocab, arch: HybridConfig, qat_cfg: QATConfig, eng_cfg: EngineConfig):
        self.vocab = vocab
        self.arch = arch
        self.cfg = eng_cfg
        self.device = torch.device(eng_cfg.device)
        self.qat = QATLoop(qat_cfg, arch, vocab_size=vocab.size)
        self.state_head = StateTransitionHead(
            StateTransitionConfig(dim=arch.dim, hidden=arch.dim * 2, n_actions=8)
        ).to(self.device)
        self.state_proj = nn.Linear(vocab.dim + vocab.ast_dim, arch.dim, bias=False).to(self.device)
        self._closed = False

    @staticmethod
    def _frame_text(frame: Frame) -> str:
        parts = []
        for ev in frame.events:
            if ev.text:
                parts.append(ev.text)
            elif ev.call is not None:
                parts.append(ev.call.name)
        return "\n".join(parts)

    def step_frame(self, frame: Frame) -> StepResult:
        text = self._frame_text(frame)
        out = encode(text, self.vocab, parse_ast=True)
        ids = out.ids
        if len(ids) < 2:
            ids = ids + ids + [0]
        seq_len = max(2, self.cfg.seq_len)
        if len(ids) > seq_len:
            ids = ids[:seq_len]
        t = torch.tensor(ids, dtype=torch.long, device=self.device)
        inp = t[:-1].unsqueeze(0)
        tgt = t[1:].unsqueeze(0)
        qat_res = self.qat.step(inp, tgt, source=frame.source)
        with torch.no_grad():
            ctx = torch.zeros(1, self.arch.dim, device=self.device)
            emb = embed(out, self.vocab)
            state = self.state_proj(torch.tensor(emb, device=self.device).mean(dim=0, keepdim=True))
            sim = self.state_head.simulate(state, ctx)
            state_loss = float(sim["action_pred"].float().mean().item())
        return StepResult(qat=qat_res, state_loss=state_loss)

    def simulate_action(self, text: str) -> dict:
        if not text:
            raise ValueError("empty action text")
        out = encode(text, self.vocab, parse_ast=True)
        emb = embed(out, self.vocab)
        state = self.state_proj(torch.tensor(emb, device=self.device).mean(dim=0, keepdim=True))
        ctx = torch.zeros(1, self.arch.dim, device=self.device)
        sim = self.state_head.simulate(state, ctx)
        return {"delta": sim["delta"], "exit_pred": sim["exit_pred"], "action_pred": sim["action_pred"]}

    def close(self) -> None:
        self._closed = True
        self.qat.close()
