"""Ingest ccsniff NDJSON session exports into verified Tianji frames."""
from __future__ import annotations

import json
from typing import Iterable, Iterator, List, Optional

from ..protocol import (
    DiffHunk,
    Frame,
    ToolCall,
    ToolResult,
    Trajectory,
    frame_hash,
    make_frame,
    verify_frame,
)


def row_to_trajectory(row: dict, sid: Optional[str] = None) -> Trajectory:
    role = row.get("role")
    rtype = row.get("type")
    text = row.get("text") or ""
    ts = int(row.get("ts", 0))
    trace = sid or row.get("sid") or "trace"
    source = "cursor"

    if rtype == "system" or role == "system":
        return Trajectory(kind="system_prompt", trace=trace, source=source, ts=ts,
                          text=f"<system>{text}</system>")
    if role == "user" and rtype == "text":
        return Trajectory(kind="context", trace=trace, source=source, ts=ts, text=text)
    if role == "assistant" and rtype == "thinking":
        return Trajectory(kind="cot", trace=trace, source=source, ts=ts, text=text)
    if role == "assistant" and rtype == "tool_use":
        try:
            args = json.loads(text) if text else {}
        except (json.JSONDecodeError, TypeError):
            args = {}
        return Trajectory(kind="tool_call", trace=trace, source=source, ts=ts,
                          call=ToolCall(name=row.get("tool") or "tool", args=args))
    if role == "tool_result" or rtype == "tool_result":
        exit_code = 0 if not row.get("isError") else 1
        return Trajectory(kind="tool_result", trace=trace, source=source, ts=ts,
                          result=ToolResult(exit=exit_code, stdout=text, stderr=None, diff=None, diff_paths=None))
    if rtype == "result" or role == "result":
        return Trajectory(kind="exec_trace", trace=trace, source=source, ts=ts,
                          duration_ms=int(row.get("duration", 0) or 0))
    if role == "assistant" and rtype == "text":
        return Trajectory(kind="context", trace=trace, source=source, ts=ts, text=text)
    return Trajectory(kind="context", trace=trace, source=source, ts=ts, text=text)


def parse_ccsniff_ndjson(lines: Iterable[str]) -> Iterator[dict]:
    for line in lines:
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)


def _trace_end(trace: str, ts: int) -> Trajectory:
    return Trajectory(kind="trace_end", trace=trace, source="cursor", ts=ts, text=None)


def rows_to_frames(rows: Iterable[dict], batch_size: int = 32) -> Iterator[Frame]:
    batch: List[Trajectory] = []
    seq = 0
    cur_trace: Optional[str] = None
    cur_ts = 0
    for row in rows:
        ev = row_to_trajectory(row, row.get("sid"))
        if cur_trace is None:
            cur_trace = ev.trace
        batch.append(ev)
        cur_ts = max(cur_ts, ev.ts)
        if len(batch) >= batch_size:
            evs = tuple(batch) + (_trace_end(cur_trace, cur_ts),)
            yield make_frame(cur_trace, "cursor", seq, evs)
            seq += 1
            batch = []
            cur_trace = None
    if batch:
        evs = tuple(batch) + (_trace_end(cur_trace or "trace", cur_ts),)
        yield make_frame(cur_trace or "trace", "cursor", seq, evs)


def ingest_ccsniff_stream(lines: Iterable[str], batch_size: int = 32) -> Iterator[Frame]:
    rows = parse_ccsniff_ndjson(lines)
    for f in rows_to_frames(rows, batch_size=batch_size):
        yield f
