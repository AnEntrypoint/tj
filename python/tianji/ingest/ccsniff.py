"""Ingest ccsniff NDJSON session exports into verified Tianji frames.

The ccsniff `--json` contract emits one NDJSON row per agent event with the
fields: {ts, iso, sid, parent, cwd, project, role, type, tool, isMeta, text}.
We map each row to a Trajectory event, group events by session (`sid`), and
emit hash-chained Frames terminated by a trace_end marker.
"""
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


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def row_to_trajectory(row: dict, sid: Optional[str] = None) -> Trajectory:
    role = row.get("role")
    rtype = row.get("type")
    text = row.get("text") or ""
    ts = _safe_int(row.get("ts"), 0)
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
        # ccsniff --json does not emit an exit code; default to success and
        # keep the mapping crash-free for any row shape.
        exit_code = 0 if not row.get("isError") else 1
        return Trajectory(kind="tool_result", trace=trace, source=source, ts=ts,
                          result=ToolResult(exit=exit_code, stdout=text, stderr=None, diff=None, diff_paths=None))
    if rtype == "result" or role == "result":
        return Trajectory(kind="exec_trace", trace=trace, source=source, ts=ts,
                          duration_ms=_safe_int(row.get("duration"), 0))
    if role == "assistant" and rtype == "text":
        return Trajectory(kind="context", trace=trace, source=source, ts=ts, text=text)
    return Trajectory(kind="context", trace=trace, source=source, ts=ts, text=text)


def parse_ccsniff_ndjson(lines: Iterable[str]) -> Iterator[dict]:
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def _trace_end(trace: str, ts: int) -> Trajectory:
    return Trajectory(kind="trace_end", trace=trace, source="cursor", ts=ts, text=None)


def rows_to_frames(rows: Iterable[dict], batch_size: int = 32) -> Iterator[Frame]:
    """Group events into frames, one frame per session chunk.

    Events are bucketed by session id (`sid`) so a frame never mixes multiple
    sessions. Within a session, events are chunked at `batch_size` and each
    emitted frame is terminated by a trace_end marker.
    """
    batch: List[Trajectory] = []
    cur_sid: Optional[str] = None
    cur_ts = 0
    seq = 0

    def flush() -> None:
        nonlocal batch, cur_sid, cur_ts, seq
        if not batch:
            return
        evs = tuple(batch) + (_trace_end(cur_sid or "trace", cur_ts),)
        yield make_frame(cur_sid or "trace", "cursor", seq, evs)
        seq += 1
        batch = []

    for row in rows:
        try:
            ev = row_to_trajectory(row, row.get("sid"))
        except Exception:
            continue
        sid = ev.trace
        if cur_sid is None:
            cur_sid = sid
        elif sid != cur_sid:
            yield from flush()
            cur_sid = sid
        batch.append(ev)
        cur_ts = max(cur_ts, ev.ts)
        if len(batch) >= batch_size:
            yield from flush()

    yield from flush()


def ingest_ccsniff_stream(lines: Iterable[str], batch_size: int = 32) -> Iterator[Frame]:
    rows = parse_ccsniff_ndjson(lines)
    for f in rows_to_frames(rows, batch_size=batch_size):
        yield f
