import torch
import pytest
from tianji import Engine, EngineConfig
from tianji.tokens.apt import Vocab
from tianji.arch.hybrid import HybridConfig
from tianji.distill.qat_loop import QATConfig
from tianji.protocol import Trajectory, ToolCall, ToolResult, DiffHunk, Frame, frame_hash


def _build_small_vocab() -> Vocab:
    lines = [
        "<tool_call>{\"name\":\"edit\",\"args\":{\"path\":\"a.py\"}}</tool_call>",
        "<bash_output>ok</bash_output>",
        "<system>agent</system>",
        "<cot>plan</cot>",
        "<diff>--- a\n+++ b\n</diff>",
        "def fib(n): return n",
    ] * 4
    return Vocab.build(lines, target_size=128, dim=16, ast_dim=4)


def test_engine_constructs():
    eng = Engine(_build_small_vocab(), HybridConfig(dim=16, n_layers=27),
                 QATConfig(device="cpu", lora_rank=4, vram_bytes=4 * 1024 ** 3),
                 EngineConfig(device="cpu", seq_len=8, batch_size=1))
    eng.close()


def test_engine_step_frame_runs():
    eng = Engine(_build_small_vocab(), HybridConfig(dim=16, n_layers=27),
                 QATConfig(device="cpu", lora_rank=4, vram_bytes=4 * 1024 ** 3),
                 EngineConfig(device="cpu", seq_len=4, batch_size=1))
    evs = (
        Trajectory(kind="system_prompt", trace="t", source="synthetic", ts=1, text="<system>agent</system>"),
        Trajectory(kind="tool_call", trace="t", source="synthetic", ts=2,
                   call=ToolCall(name="edit", args={"path": "x.py"}, args_ast=None)),
        Trajectory(kind="tool_result", trace="t", source="synthetic", ts=3,
                   result=ToolResult(exit=0, stdout="ok", stderr=None, diff=None, diff_paths=None)),
        Trajectory(kind="diff", trace="t", source="synthetic", ts=4,
                   hunks=(DiffHunk(path="x.py", added=1, removed=0, text="+new"),)),
    )
    f = Frame(trace="t", source="synthetic", seq=0, events=evs, hash=frame_hash("t", "synthetic", 0, evs))
    res = eng.step_frame(f)
    assert res is not None
    assert hasattr(res, "qat") and hasattr(res, "state_loss")
    assert res.qat.loss > 0
    assert res.qat.kd_loss >= 0
    eng.close()


def test_engine_simulate_action_returns_predictions():
    eng = Engine(_build_small_vocab(), HybridConfig(dim=16, n_layers=27),
                 QATConfig(device="cpu", lora_rank=4, vram_bytes=4 * 1024 ** 3),
                 EngineConfig(device="cpu", seq_len=4, batch_size=1))
    sim = eng.simulate_action("<tool_call>{\"name\":\"edit\"}</tool_call>")
    assert sim["delta"].shape == (1, 16)
    assert sim["exit_pred"].dtype == torch.long
    eng.close()


def test_engine_simulate_action_rejects_empty():
    eng = Engine(_build_small_vocab(), HybridConfig(dim=16, n_layers=27),
                 QATConfig(device="cpu", lora_rank=4, vram_bytes=4 * 1024 ** 3),
                 EngineConfig(device="cpu", seq_len=4, batch_size=1))
    with pytest.raises(ValueError, match="empty"):
        eng.simulate_action("")
    eng.close()
