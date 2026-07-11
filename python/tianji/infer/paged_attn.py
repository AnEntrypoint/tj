"""Paged KV cache with block-level allocation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

BLOCK_TOKENS = 16


@dataclass
class PagedKVCache:
    blocks: int

    @classmethod
    def open(cls, blocks: int) -> "PagedKVCache":
        return cls(blocks=blocks)

    def __post_init__(self):
        self.free: List[int] = list(range(self.blocks))
        self.sequences: Dict[int, List[int]] = {}
        self._next_seq = 0

    def alloc_sequence(self) -> int:
        sid = self._next_seq
        self._next_seq += 1
        self.sequences[sid] = []
        return sid

    def extend(self, sid: int, n_tokens: int) -> None:
        needed = (n_tokens + BLOCK_TOKENS - 1) // BLOCK_TOKENS
        for _ in range(needed):
            if not self.free:
                raise MemoryError("KV cache exhausted")
            self.sequences[sid].append(self.free.pop())

    def free_sequence(self, sid: int) -> None:
        for b in self.sequences.pop(sid, []):
            self.free.append(b)
