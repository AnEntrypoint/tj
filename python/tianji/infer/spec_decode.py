"""Speculative decoding step using an MTP head."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SpeculativeResult:
    total: int
    accepted: int
    next_token: torch.Tensor


def speculative_step(embed, model, head, mtp, ctx: torch.Tensor) -> SpeculativeResult:
    h = embed(ctx)
    h, _, _ = model(h)
    last = h[:, -1:, :]
    logits = head(last)
    draft = logits[-1].argmax(dim=-1)
    tk = mtp.speculate(last.squeeze(1))
    n = len(tk)
    accepted = int((torch.stack(tk) == draft.expand(n)).sum().item())
    return SpeculativeResult(total=n, accepted=accepted, next_token=tk[-1])
