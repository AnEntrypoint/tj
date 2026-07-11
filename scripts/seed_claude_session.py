#!/usr/bin/env python3
"""Synthesize Claude Code session JSONL under ~/.claude/projects/."""
from __future__ import annotations
import json
import time
from pathlib import Path

ROOT = Path.home() / ".claude" / "projects"
PROJECT = "tianji-demo"
SESSION_ID = "demo-session-0001"
OUT = ROOT / PROJECT / f"{SESSION_ID}.jsonl"


def _ts(offset_ms: int) -> str:
    base = 1700000000000
    return time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime((base + offset_ms) / 1000.0))


def _user_text(text, ts_ms, sid, cwd):
    return {"type": "user", "sessionId": sid, "timestamp": _ts(ts_ms), "cwd": cwd,
            "gitBranch": "main", "message": {"role": "user", "content": text}}


def _user_meta(text, ts_ms, sid, cwd):
    return {"type": "user", "sessionId": sid, "timestamp": _ts(ts_ms), "cwd": cwd,
            "gitBranch": "main", "isMeta": True, "message": {"role": "user", "content": text}}


def _assistant_blocks(blocks, ts_ms, sid, cwd, msg_id="msg_001", stop_reason=None):
    return {"type": "assistant", "sessionId": sid, "timestamp": _ts(ts_ms), "cwd": cwd,
            "gitBranch": "main",
            "message": {"id": msg_id, "role": "assistant", "content": blocks, "stop_reason": stop_reason}}


def _tool_result_block(tool_use_id, content, is_error=False):
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content, "is_error": is_error}


def _user_tool_result(block, ts_ms, sid, cwd):
    return {"type": "user", "sessionId": sid, "timestamp": _ts(ts_ms), "cwd": cwd,
            "gitBranch": "main", "message": {"role": "user", "content": [block]}}


def _system_init(ts_ms, sid, cwd, model="claude-3-5-sonnet"):
    return {"type": "system", "sessionId": sid, "timestamp": _ts(ts_ms), "cwd": cwd,
            "gitBranch": "main", "subtype": "init", "model": model,
            "tools": ["Read", "Edit", "Bash", "Grep"]}


def _result(ts_ms, sid, cwd, duration_ms, cost, is_error=False):
    return {"type": "result", "sessionId": sid, "timestamp": _ts(ts_ms), "cwd": cwd,
            "gitBranch": "main", "subtype": "success", "duration_ms": duration_ms,
            "total_cost_usd": cost, "is_error": is_error, "result": "Turn complete."}


def build_session():
    sid = SESSION_ID; cwd = f"/home/user/{PROJECT}"; events = []; t = 0
    events.append(_system_init(t, sid, cwd)); t += 100
    events.append(_user_meta("<system-reminder>You are Claude Code.</system-reminder>", t, sid, cwd)); t += 100
    events.append(_user_text("Read src/app.py and add a main guard.", t, sid, cwd)); t += 200
    events.append(_assistant_blocks([
        {"type": "thinking", "thinking": "Read first, then edit."},
        {"type": "tool_use", "id": "toolu_001", "name": "Read", "input": {"file_path": "src/app.py"}},
    ], t, sid, cwd, msg_id="msg_001")); t += 300
    events.append(_user_tool_result(_tool_result_block("toolu_001", "def main(): print('hi')"), t, sid, cwd)); t += 200
    events.append(_assistant_blocks([
        {"type": "thinking", "thinking": "Add main guard."},
        {"type": "tool_use", "id": "toolu_002", "name": "Edit", "input": {
            "file_path": "src/app.py",
            "old_string": "def main():\n    print('hello')\n",
            "new_string": "def main():\n    if __name__ == '__main__':\n        print('hello')\n",
        }},
    ], t, sid, cwd, msg_id="msg_002")); t += 300
    events.append(_user_tool_result(_tool_result_block("toolu_002", "Updated."), t, sid, cwd)); t += 200
    events.append(_assistant_blocks([
        {"type": "text", "text": "Added the main guard."},
    ], t, sid, cwd, msg_id="msg_003", stop_reason="end_turn")); t += 200
    events.append(_result(t, sid, cwd, duration_ms=1500, cost=0.0023)); t += 100
    return events


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    events = build_session()
    with OUT.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    print(f"wrote {len(events)} events to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
