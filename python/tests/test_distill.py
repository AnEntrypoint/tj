import torch
import torch.nn as nn
import pytest
from tianji.distill.lora import (LoRAConfig, wrap_linear_with_lora, merge_lora,
                                  save_lora_state, load_lora_state)
from tianji.distill.replay import ReplayBuffer
from tianji.distill.router_alignment import RouterAlignment, apply_router_bias
from tianji.distill.kd import kd_loss, StubTeacher
from tianji.distill.qat_loop import QATLoop, QATConfig, QATStepResult
from tianji.arch.hybrid import HybridConfig


def test_lora_initial_output_is_zero():
    lin = nn.Linear(8, 4, bias=False)
    lin.weight.data.copy_(torch.randn(4, 8))
    wrapped = wrap_linear_with_lora(lin, LoRAConfig(rank=2))
    x = torch.randn(2, 8)
    assert torch.allclose(wrapped(x), lin(x), atol=1e-6)


def test_lora_merge_into_base():
    lin = nn.Linear(8, 4, bias=False)
    lin.weight.data.copy_(torch.randn(4, 8))
    wrapped = wrap_linear_with_lora(lin, LoRAConfig(rank=2, alpha=4.0))
    with torch.no_grad():
        wrapped.lora.B.add_(torch.ones_like(wrapped.lora.B) * 0.1)
    x = torch.randn(2, 8)
    pre = wrapped(x)
    n = merge_lora(wrapped)
    assert n >= 1
    assert torch.allclose(pre, wrapped(x), atol=1e-5)


def test_lora_save_load_round_trip():
    lin = nn.Linear(8, 4, bias=False)
    wrapped_a = wrap_linear_with_lora(lin, LoRAConfig(rank=2))
    with torch.no_grad():
        wrapped_a.lora.A.add_(torch.randn_like(wrapped_a.lora.A) * 0.5)
        wrapped_a.lora.B.add_(torch.randn_like(wrapped_a.lora.B) * 0.5)
    state = save_lora_state(wrapped_a)
    lin2 = nn.Linear(8, 4, bias=False)
    wrapped_b = wrap_linear_with_lora(lin2, LoRAConfig(rank=2))
    n = load_lora_state(wrapped_b, state)
    assert n >= 1
    assert torch.allclose(wrapped_a.lora.A, wrapped_b.lora.A)
    assert torch.allclose(wrapped_a.lora.B, wrapped_b.lora.B)


def test_replay_ring_buffer_bounds_capacity():
    rb = ReplayBuffer(capacity=4)
    for i in range(10):
        rb.push(torch.tensor([i]), torch.tensor([i]))
    assert len(rb) == 4


def test_replay_sample_returns_none_when_empty():
    assert ReplayBuffer(capacity=4).sample(2) is None


def test_router_alignment_known_sources():
    ra = RouterAlignment.build(16)
    assert ra.bias_for("claude")[12] > 0
    assert ra.bias_for("gpt-4o")[15] > 0


def test_kd_loss_zero_on_identical_logits():
    logits = torch.randn(2, 4, 16)
    assert kd_loss(logits, logits, T=2.0).item() < 1e-6


def test_stub_teacher_is_deterministic():
    teacher = StubTeacher(vocab_size=32)
    ids = torch.randint(0, 32, (2, 5))
    assert torch.equal(teacher(ids), teacher(ids))


def test_qat_loop_runs_step():
    cfg = QATConfig(device="cpu", lora_rank=4, vram_bytes=4 * 1024 ** 3)
    qat = QATLoop(cfg, HybridConfig(dim=16, n_layers=27), vocab_size=64)
    res = qat.step(torch.randint(0, 64, (2, 8)), torch.randint(0, 64, (2, 8)), source="cursor")
    assert isinstance(res, QATStepResult)
    assert res.loss > 0
    assert res.kd_loss >= 0
    assert res.vram_used_bytes > 0
    qat.close()


def test_qat_loop_checkpoint_save_load(tmp_path):
    cfg = QATConfig(device="cpu", lora_rank=4, vram_bytes=4 * 1024 ** 3)
    qat = QATLoop(cfg, HybridConfig(dim=16, n_layers=27), vocab_size=64)
    qat.step(torch.randint(0, 64, (2, 8)), torch.randint(0, 64, (2, 8)), source="synthetic")
    path = str(tmp_path / "ckpt.pt")
    qat.save_checkpoint(path)
    qat2 = QATLoop(cfg, HybridConfig(dim=16, n_layers=27), vocab_size=64)
    assert qat2.load_checkpoint(path) == 1
    qat.close(); qat2.close()
