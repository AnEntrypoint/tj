import torch
import torch.nn as nn
import pytest
from tianji.quant.fakequant import (FakeQuantLinear, fakequant_int4, apply_fakequant_to_linear,
                                     materialize_real_int4)
from tianji.quant.kv_quant import quant_int2, dequant_int2, pack_int2, unpack_int2, INT2_MAX, INT2_MIN
from tianji.quant.adam8bit import Adam8bit


def test_fakequant_is_straight_through():
    x = torch.randn(4, 8, requires_grad=True)
    fakequant_int4(x).sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x.grad))


def test_fakequant_linear_inherits_nn_linear():
    fq = FakeQuantLinear(8, 4)
    assert isinstance(fq, nn.Linear)
    assert fq(torch.randn(2, 8)).shape == (2, 4)


def test_int2_round_trip_within_levels():
    x = torch.tensor([[-1.0, -0.5, 0.0, 0.5, 1.0]])
    q, s = quant_int2(x)
    assert q.min() >= INT2_MIN
    assert q.max() <= INT2_MAX
    assert dequant_int2(q, s).shape == x.shape


def test_int2_pack_unpack_round_trip():
    q = torch.tensor([[-2, -1, 0, 1, 1, 0, -1, -2]] * 2, dtype=torch.int8)
    packed = pack_int2(q)
    assert torch.equal(unpack_int2(packed, q.numel()).reshape(q.shape), q)


def test_adam8bit_runs_and_reduces_loss():
    torch.manual_seed(0)
    model = nn.Linear(8, 4)
    opt = Adam8bit(model.parameters(), lr=0.1)
    target = torch.randn(2, 4)
    losses = []
    for _ in range(20):
        y = model(torch.randn(2, 8))
        loss = ((y - target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_adam8bit_state_is_int8():
    model = nn.Linear(4, 2)
    opt = Adam8bit(model.parameters(), lr=0.01)
    model(torch.randn(1, 4)).sum().backward()
    opt.step()
    for p in model.parameters():
        assert opt.state[p]["m_q"].dtype == torch.int8
        assert opt.state[p]["v_q"].dtype == torch.int8
