import json
import pytest
from tianji.ingest.ccsniff import (row_to_trajectory, rows_to_frames,
                                    parse_ccsniff_ndjson, ingest_ccsniff_stream)
from tianji.protocol import verify_frame


SAMPLE_ROWS = [
    {"ts": 1000, "sid": "s1", "role": "system", "type": "system",
     "text": "You are Claude Code.", "tool": None, "isMeta": False, "isError": False},
    {"ts": 1100, "sid": "s1", "role": "user", "type": "text", "isMeta": True,
     "text": "<system-reminder>Use tools.</system-reminder>", "tool": None, "isError": False},
    {"ts": 1200, "sid": "s1", "role": "user", "type": "text", "isMeta": False,
     "text": "Read src/app.py", "tool": None, "isError": False},
    {"ts": 1300, "sid": "s1", "role": "assistant", "type": "thinking",
     "text": "I need to read the file.", "tool": None, "isMeta": False, "isError": False},
    {"ts": 1400, "sid": "s1", "role": "assistant", "type": "tool_use",
     "text": '{"file_path":"src/app.py"}', "tool": "Read", "isMeta": False, "isError": False},
    {"ts": 1500, "sid": "s1", "role": "tool_result", "type": "tool_result",
     "text": "def main(): print('hi')", "tool": None, "isMeta": False, "isError": False},
    {"ts": 1600, "sid": "s1", "role": "assistant", "type": "text",
     "text": "Done.", "tool": None, "isMeta": False, "isError": False},
    {"ts": 1700, "sid": "s1", "role": "result", "type": "result",
     "text": "", "tool": None, "isMeta": False, "isError": False, "duration": 500},
]


def test_row_system_to_system_prompt():
    ev = row_to_trajectory(SAMPLE_ROWS[0], "s1")
    assert ev.kind == "system_prompt"
    assert "<system>" in ev.text


def test_row_user_text_to_context():
    ev = row_to_trajectory(SAMPLE_ROWS[2], "s1")
    assert ev.kind == "context"
    assert ev.text == "Read src/app.py"


def test_row_thinking_to_cot():
    ev = row_to_trajectory(SAMPLE_ROWS[3], "s1")
    assert ev.kind == "cot"


def test_row_tool_use_to_tool_call():
    ev = row_to_trajectory(SAMPLE_ROWS[4], "s1")
    assert ev.kind == "tool_call"
    assert ev.call.name == "Read"
    assert ev.call.args == {"file_path": "src/app.py"}


def test_row_tool_result_to_tool_result():
    ev = row_to_trajectory(SAMPLE_ROWS[5], "s1")
    assert ev.kind == "tool_result"
    assert ev.result.exit == 0


def test_row_result_to_exec_trace():
    ev = row_to_trajectory(SAMPLE_ROWS[7], "s1")
    assert ev.kind == "exec_trace"
    assert ev.duration_ms == 500


def test_rows_to_frames_emits_hash_verified_frames():
    frames = list(rows_to_frames(iter(SAMPLE_ROWS), batch_size=4))
    for f in frames:
        assert verify_frame(f)
        assert f.source == "cursor"


def test_rows_to_frames_inserts_trace_end():
    frames = list(rows_to_frames(iter(SAMPLE_ROWS), batch_size=100))
    assert len(frames) == 1
    assert frames[0].events[-1].kind == "trace_end"


def test_ingest_ccsniff_stream_end_to_end():
    lines = [json.dumps(r) for r in SAMPLE_ROWS]
    frames = list(ingest_ccsniff_stream(iter(lines), batch_size=32))
    assert len(frames) == 1
    f = frames[0]
    assert verify_frame(f)
    assert len(f.events) == 9  # 8 source + 1 trace_end
    kinds = [e.kind for e in f.events]
    assert "system_prompt" in kinds
    assert "tool_call" in kinds
    assert "trace_end" in kinds
