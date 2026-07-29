"""Token generator driving the QAT model with incremental Mamba-2 state.

Pre-allocates a fixed-size buffer so the autoregressive loop never calls
``torch.cat``, eliminating O(N) allocation per token. Uses Mamba-2 state
carry for incremental decoding: the full prompt is processed once, then
each subsequent token is fed individually with the carried state, making
inference ~10x faster for long sequences.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

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
        self._buf: torch.Tensor | None = None
        # Mamba-2 state cache for incremental decoding.
        self._mamba_state: Tuple[torch.Tensor, ...] | None = None

    def _ensure_buf(self, n: int) -> torch.Tensor:
        if self._buf is None or n > self._buf.shape[1]:
            self._buf = torch.zeros(
                1, n + self.cfg.max_tokens, dtype=torch.long, device=self.device)
        return self._buf

    def generate(self, prompt_ids) -> Iterator[GenerateStep]:
        if not isinstance(prompt_ids, torch.Tensor):
            prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
        ids = prompt_ids.to(self.device).unsqueeze(0)
        plen = ids.shape[1]
        buf = self._ensure_buf(plen)
        buf[:, :plen].copy_(ids)
        cur_len = plen

        # ── First step: full forward on the prompt, cache Mamba-2 state ──
        with torch.no_grad():
            h = self.model.embed(buf[:, :cur_len])
            h, _, next_state = self.model.stack(h)
            logits = self.model.head(h)
        nxt = int(logits[0, -1].argmax(dim=-1).item())
        yield GenerateStep(token=nxt)
        buf[0, cur_len] = nxt
        cur_len += 1
        self._mamba_state = next_state

        # ── Incremental steps: feed only the last token with cached state ──
        for _ in range(1, self.cfg.max_tokens):
            last = buf[:, cur_len - 1:cur_len]  # (1, 1)
            with torch.no_grad():
                h = self.model.embed(last)
                h, _, next_state = self.model.stack(h, state_tuple=self._mamba_state)
                logits = self.model.head(h)
            nxt = int(logits[0, -1].argmax(dim=-1).item())
            yield GenerateStep(token=nxt)
            buf[0, cur_len] = nxt
            cur_len += 1
            self._mamba_state = next_state

    def reset_state(self) -> None:
        """Clear cached Mamba-2 state for a new generation."""
        self._mamba_state = None