"""Capability and resource-budget primitives.

Caps are unforgeable tokens minted via `grant`. Regions are affine handles
that can be closed exactly once. ResourceBudget tracks a fixed pool.
"""
from __future__ import annotations

import threading
import uuid

_KIND_LOCK = threading.Lock()


class Cap:
    __slots__ = ("_kind", "_rid")

    def __init__(self, kind: str, rid: str):
        self._kind = kind
        self._rid = rid

    @property
    def kind(self) -> str:
        return self._kind

    def __eq__(self, other):
        return isinstance(other, Cap) and self._kind == other._kind and self._rid == other._rid

    def __hash__(self):
        return hash((self._kind, self._rid))

    def __repr__(self):
        return f"Cap({self._kind}:{self._rid[:8]})"


def grant(kind: str) -> Cap:
    rid = uuid.uuid4().hex
    return Cap(kind, rid)


def assert_cap(cap: Cap, kind: str) -> None:
    if not isinstance(cap, Cap) or cap.kind != kind:
        raise PermissionError(f"cap kind {getattr(cap, 'kind', None)!r} != required {kind!r}")


class Region:
    __slots__ = ("_rid", "_kind", "_value", "_closed", "_lock")

    def __init__(self, kind: str, rid: str, value):
        self._rid = rid
        self._kind = kind
        self._value = value
        self._closed = False
        self._lock = threading.Lock()

    @classmethod
    def open(cls, kind: str, value) -> "Region":
        return cls(kind, uuid.uuid4().hex, value)

    @property
    def rid(self) -> str:
        return self._rid

    @property
    def value(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("region already closed")
            return self._value

    def close(self) -> None:
        with self._lock:
            self._closed = True

    def __repr__(self):
        return f"Region({self._kind})"


def disjoint(a: Region, b: Region) -> bool:
    return a.rid != b.rid


class ResourceBudget:
    def __init__(self, name: str, capacity_bytes: int):
        self.name = name
        self.capacity_bytes = int(capacity_bytes)
        self.used_bytes = 0
        self._lock = threading.Lock()

    @property
    def remaining(self) -> int:
        return self.capacity_bytes - self.used_bytes

    def allocate(self, n_bytes: int) -> None:
        n_bytes = int(n_bytes)
        if n_bytes < 0:
            raise ValueError("allocate negative")
        with self._lock:
            if self.used_bytes + n_bytes > self.capacity_bytes:
                raise MemoryError(f"{self.name} budget exhausted: {self.used_bytes}+{n_bytes} > {self.capacity_bytes}")
            self.used_bytes += n_bytes

    def free(self, n_bytes: int) -> None:
        n_bytes = int(n_bytes)
        with self._lock:
            self.used_bytes = max(0, self.used_bytes - n_bytes)
