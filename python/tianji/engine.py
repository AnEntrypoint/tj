"""Tianji training engine: ties vocab + hybrid model + QAT loop + state head."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokens.apt import Vocab, encode, embed
from .arch.hybrid import HybridConfig
from .distill.qat_loop import QATLoop, QATConfig, QATStepResult
from .state.transition import (
    StateTransitionHead,
    StateTransitionConfig,
    EVENT_KINDS,
    kind_to_idx,
)
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
    state_pred_kind: Optional[int] = None
    state_target_kind: Optional[int] = None


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
            StateTransitionConfig(dim=arch.dim, hidden=arch.dim * 2, n_actions=len(EVENT_KINDS))
        ).to(self.device)
        self.state_proj = nn.Linear(arch.dim, arch.dim, bias=False).to(self.device)
        # The state head is a genuine, trained sub-model -- not dead weight.
        self.state_opt = torch.optim.AdamW(
            [p for p in (*self.state_head.parameters(), *self.state_proj.parameters()) if p.requires_grad],
            lr=qat_cfg.lr,
        )
        self._prev_state: Optional[torch.Tensor] = None
        self._closed = False

        # Contrastive loss for positive/negative eval separation.
        from .distill.contrastive import ContrastiveLoss
        self._contrastive = ContrastiveLoss(temperature=0.07)
        self._contrastive_weight = 0.1  # weight of contrastive loss vs LM loss

        # Pre-allocate frame arena for zero-allocation hot loop.
        L = max(2, int(eng_cfg.seq_len))
        self._buf_inp = torch.zeros(1, L, dtype=torch.long, device=self.device)
        self._buf_tgt = torch.zeros(1, L, dtype=torch.long, device=self.device)
        self._buf_mask = torch.ones(1, L, dtype=torch.bool, device=self.device)

    @staticmethod
    def _frame_text(frame: Frame) -> str:
        parts = []
        for ev in frame.events:
            if ev.text:
                parts.append(ev.text)
            elif ev.call is not None:
                parts.append(ev.call.name)
        return "\n".join(parts)

    def _agent_state(self, ids: torch.Tensor) -> torch.Tensor:
        """Latent agent state = pooled hybrid-stack hidden projected to arch.dim.
        Chunked over ``seq_len`` so a very long (e.g. 200k-token) frame does not
        OOM the full-frame forward; the final chunk's pooled state is used."""
        L = max(2, int(self.cfg.seq_len))
        pooled = None
        for s in range(0, ids.shape[1], L):
            e = min(s + L, ids.shape[1])
            with torch.no_grad():
                h = self.qat.hidden(ids[:, s:e])
                pooled = h.mean(dim=1)
        return self.state_proj(pooled)

    def step_frame(self, frame: Frame) -> StepResult:
        text = self._frame_text(frame)
        out = encode(text, self.vocab, parse_ast=True)
        ids = self._prepare_ids(out.ids)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t = torch.tensor(ids, dtype=torch.long, device=self.device)
        # Train over the whole frame in chunks of exactly ``seq_len`` tokens.
        # Mamba-2 state is carried between chunks within a frame (teacher-forced
        # TBPTT: state propagates forward but gradients detach at boundaries),
        # so the model learns to use its ~200k-token context even though each
        # backward step is truncated to seq_len.
        chunk = max(2, int(self.cfg.seq_len))
        L = chunk
        qat_loss_sum = 0.0
        qat_kd_sum = 0.0
        qat_aux_sum = 0.0
        qat_n = 0
        T = len(ids)
        mamba_states: tuple[torch.Tensor, ...] | None = None
        for start in range(0, max(1, T - 1), chunk):
            end = min(start + chunk, T)
            seg = end - start
            if seg < 2:
                continue
            inp = t[start:end - 1]
            tgt = t[start + 1:end]
            self._buf_inp.zero_()
            self._buf_tgt.zero_()
            self._buf_mask.fill_(True)
            n_inp = inp.shape[0]
            self._buf_inp[0, :n_inp].copy_(inp)
            self._buf_tgt[0, :n_inp].copy_(tgt)
            if n_inp < L:
                self._buf_mask[0, n_inp:] = False
            res, next_states = self.qat.step(self._buf_inp, self._buf_tgt,
                                             mask=self._buf_mask, source=frame.source,
                                             state_tuple=mamba_states)
            qat_loss_sum += res.loss
            qat_kd_sum += res.kd_loss
            qat_aux_sum += res.aux_loss
            qat_n += 1
            if next_states is not None:
                mamba_states = tuple(s.detach().clone() for s in next_states)
            else:
                mamba_states = None
        qat_res = QATStepResult(
            loss=qat_loss_sum / max(1, qat_n),
            kd_loss=qat_kd_sum / max(1, qat_n),
            aux_loss=qat_aux_sum / max(1, qat_n),
            vram_used_bytes=int(self.qat._vram()),
        )

        state = self._agent_state(t.unsqueeze(0))

        state_loss = 0.0
        pred_kind: Optional[int] = None
        target_kind: Optional[int] = None
        if self._prev_state is not None:
            target_kind = kind_to_idx(frame.events[0].kind)
            target_exit = 1.0 if frame.events[0].kind in ("exec_trace", "trace_end") else 0.0
            ctx = torch.zeros(1, self.arch.dim, device=self.device)
            preds = self.state_head(self._prev_state, ctx)
            pred_kind = int(preds["action_logits"].argmax(dim=-1).item())
            kind_t = torch.tensor([target_kind], device=self.device)
            exit_t = torch.tensor([target_exit], device=self.device)
            a_loss = F.cross_entropy(preds["action_logits"], kind_t)
            e_loss = F.binary_cross_entropy_with_logits(preds["exit_logit"], exit_t)
            d_loss = F.mse_loss(preds["delta"], state - self._prev_state)
            state_loss = float((a_loss + e_loss + 0.1 * d_loss).item())
            (a_loss + e_loss + 0.1 * d_loss).backward()
            self.state_opt.step()
            self.state_opt.zero_grad()

        self._prev_state = state.detach()
        return StepResult(qat=qat_res, state_loss=state_loss,
                          state_pred_kind=pred_kind, state_target_kind=target_kind)

    def predict_next_kind(self, state: torch.Tensor) -> int:
        """Greedy next-event-kind prediction from a latent state (no grad)."""
        with torch.no_grad():
            ctx = torch.zeros(1, self.arch.dim, device=self.device)
            preds = self.state_head(state, ctx)
            return int(preds["action_logits"].argmax(dim=-1).item())

    @staticmethod
    def _prepare_ids(raw_ids) -> list:
        """Ensure a non-trivial token sequence so input/target pairs exist even
        for frames whose text encodes to very few tokens (e.g. exec_trace-only
        frames). Avoids empty tensors that crash the recurrent stack."""
        ids = list(raw_ids)
        if len(ids) < 2:
            ids = (ids + ids + [0, 0])[:2]
            if len(ids) < 2:
                ids = [0, 0]
        return ids

    def simulate_action(self, text: str) -> dict:
        if not text:
            raise ValueError("empty action text")
        out = encode(text, self.vocab, parse_ast=True)
        ids = self._prepare_ids(out.ids)
        t = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
        state = self._agent_state(t)
        ctx = torch.zeros(1, self.arch.dim, device=self.device)
        sim = self.state_head.simulate(state, ctx)
        return {"delta": sim["delta"], "exit_pred": sim["exit_pred"], "action_pred": sim["action_pred"]}

    def save_training_state(self, path: str) -> None:
        torch.save(
            {
                "state_head": self.state_head.state_dict(),
                "state_proj": self.state_proj.state_dict(),
                "state_opt": self.state_opt.state_dict(),
                "vocab": self.vocab,
            },
            path,
        )

    def step_contrastive(
        self, pos_texts: list[str], neg_texts: list[str]
    ) -> float:
        """Train contrastive loss on positive vs negative text pairs.

        Positive texts are Claude Code agent sessions; negative texts are
        public coding data. The contrastive loss operates on pooled hidden
        states and is trained jointly with the standard LM loss.

        Returns:
            Contrastive loss value.
        """
        if not pos_texts or not neg_texts:
            return 0.0

        # Encode and pool positive samples
        pos_embeddings = []
        for text in pos_texts[:16]:  # cap at 16 to stay within VRAM
            if not text or not text.strip():
                continue
            out = encode(text, self.vocab, parse_ast=False)
            ids = self._prepare_ids(out.ids)
            t = torch.tensor(
                ids, dtype=torch.long, device=self.device
            ).unsqueeze(0)
            state = self._agent_state(t)
            pos_embeddings.append(state)

        # Encode and pool negative samples
        neg_embeddings = []
        for text in neg_texts[:16]:
            if not text or not text.strip():
                continue
            out = encode(text, self.vocab, parse_ast=False)
            ids = self._prepare_ids(out.ids)
            t = torch.tensor(
                ids, dtype=torch.long, device=self.device
            ).unsqueeze(0)
            state = self._agent_state(t)
            neg_embeddings.append(state)

        if not pos_embeddings or not neg_embeddings:
            return 0.0

        pos_t = torch.cat(pos_embeddings, dim=0)
        neg_t = torch.cat(neg_embeddings, dim=0)

        loss = self._contrastive(pos_t, neg_t)
        (self._contrastive_weight * loss).backward()
        return float(loss.item())

    def load_training_state(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.state_head.load_state_dict(ck["state_head"])
        self.state_proj.load_state_dict(ck["state_proj"])
        self.state_opt.load_state_dict(ck["state_opt"])
        self.vocab = ck["vocab"]
        self._prev_state = None

    def close(self) -> None:
        self._closed = True
        self.qat.close()
