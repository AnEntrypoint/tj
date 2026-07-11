"""Ring attention: chunk a sequence with overlap for ring-wise attention."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RingConfig:
    ring_size: int = 4
    overlap: int = 1


def ring_chunk(x: torch.Tensor, cfg: RingConfig):
    b, n, d = x.shape
    size = cfg.ring_size
    overlap = cfg.overlap
    chunks = []
    start = 0
    first = True
    while start < n:
        if first:
            end = min(start + size, n)
            first = False
        else:
            end = min(start + size + overlap, n)
        chunks.append(x[:, start:end, :])
        if end >= n:
            break
        start = end - overlap
    return chunks


def ring_attn_forward(chunks, attn_fn):
    outs = []
    for i, c in enumerate(chunks):
        window = c
        if i > 0:
            prev = chunks[i - 1]
            keep = min(prev.shape[1], 1)
            window = torch.cat([prev[:, -keep:, :], window], dim=1)
        if i < len(chunks) - 1:
            nxt = chunks[i + 1]
            keep = min(nxt.shape[1], 1)
            window = torch.cat([window, nxt[:, :keep, :]], dim=1)
        outs.append(attn_fn(window, c))
    return torch.cat(outs, dim=1)
