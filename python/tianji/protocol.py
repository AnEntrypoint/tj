"""Tianji protocol: canonical framing of agent trajectories.

Frames are the unit of training data. Each frame carries a verified hash so
ingested sessions can be replayed deterministically.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Optional, Tuple

import torch

KH = "sha256:"


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_default)


def _default(o):
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    if isinstance(o, (bytes, bytearray)):
        return o.hex()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    raise TypeError(f"not serializable: {type(o)}")


@dataclasses.dataclass
class ToolCall:
    name: str
    args: dict
    args_ast: Optional[dict] = None


@dataclasses.dataclass
class ToolResult:
    exit: int
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    diff: Optional[str] = None
    diff_paths: Optional[Tuple[str, ...]] = None


@dataclasses.dataclass
class DiffHunk:
    path: str
    added: int
    removed: int
    text: str


@dataclasses.dataclass
class Trajectory:
    kind: str
    trace: str
    source: str
    ts: int
    text: Optional[str] = None
    call: Optional[ToolCall] = None
    result: Optional[ToolResult] = None
    hunks: Optional[Tuple[DiffHunk, ...]] = None
    duration_ms: Optional[int] = None


@dataclasses.dataclass
class Frame:
    trace: str
    source: str
    seq: int
    events: Tuple[Trajectory, ...]
    hash: str


def frame_hash(trace: str, source: str, seq: int, events) -> str:
    payload = canonical_json({"trace": trace, "source": source, "seq": seq, "events": list(events)})
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return KH + digest


def make_frame(trace: str, source: str, seq: int, events) -> Frame:
    events = tuple(events)
    return Frame(trace=trace, source=source, seq=seq, events=events, hash=frame_hash(trace, source, seq, events))


def verify_frame(f: Frame) -> bool:
    if not isinstance(f, Frame):
        return False
    if not f.hash.startswith(KH):
        return False
    return frame_hash(f.trace, f.source, f.seq, f.events) == f.hash


def parse_frame(obj) -> Frame:
    if isinstance(obj, str):
        obj = json.loads(obj)
    events = tuple(_traj_from_dict(e) for e in obj["events"])
    return Frame(trace=obj["trace"], source=obj["source"], seq=obj["seq"], events=events, hash=obj["hash"])


def _traj_from_dict(d: dict) -> Trajectory:
    call = d.get("call")
    if call is not None:
        call = ToolCall(name=call["name"], args=call.get("args", {}), args_ast=call.get("args_ast"))
    result = d.get("result")
    if result is not None:
        result = ToolResult(
            exit=result["exit"],
            stdout=result.get("stdout"),
            stderr=result.get("stderr"),
            diff=result.get("diff"),
            diff_paths=tuple(result["diff_paths"]) if result.get("diff_paths") else None,
        )
    hunks = d.get("hunks")
    if hunks is not None:
        hunks = tuple(DiffHunk(path=h["path"], added=h["added"], removed=h["removed"], text=h["text"]) for h in hunks)
    return Trajectory(
        kind=d["kind"],
        trace=d["trace"],
        source=d["source"],
        ts=d["ts"],
        text=d.get("text"),
        call=call,
        result=result,
        hunks=hunks,
        duration_ms=d.get("duration_ms"),
    )
