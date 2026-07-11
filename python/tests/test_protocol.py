from tianji.protocol import (parse_frame, verify_frame, frame_hash, make_frame,
                              Trajectory, ToolCall, ToolResult, DiffHunk, canonical_json)


def test_canonical_json_sorted_keys():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_frame_hash_deterministic():
    ev = Trajectory(kind="system_prompt", trace="t", source="synthetic", ts=1, text="<system>x</system>")
    h1 = frame_hash("t", "synthetic", 0, [ev])
    assert h1 == frame_hash("t", "synthetic", 0, [ev])
    assert h1.startswith("sha256:")


def test_make_frame_verify():
    ev = Trajectory(kind="tool_call", trace="t", source="synthetic", ts=1,
                    call=ToolCall(name="edit", args={"path": "x"}, args_ast=None))
    assert verify_frame(make_frame("t", "synthetic", 0, [ev]))


def test_diff_hunks_preserved():
    ev = Trajectory(kind="diff", trace="t", source="synthetic", ts=1,
                    hunks=(DiffHunk(path="x.py", added=2, removed=1, text="+new\n-old"),))
    f = make_frame("t", "synthetic", 0, [ev])
    assert verify_frame(f)
    assert f.events[0].hunks[0].added == 2
