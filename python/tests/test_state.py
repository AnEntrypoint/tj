import torch
from tianji.state.transition import StateTransitionHead, StateTransitionConfig


def test_forward_shapes():
    cfg = StateTransitionConfig(dim=16, hidden=32, n_actions=8)
    head = StateTransitionHead(cfg)
    out = head(torch.randn(2, 16), torch.randn(2, 16))
    assert out["delta"].shape == (2, 16)
    assert out["exit_logit"].shape == (2,)
    assert out["action_logits"].shape == (2, 8)


def test_simulate_no_grad():
    head = StateTransitionHead(StateTransitionConfig(dim=16, hidden=32, n_actions=8))
    sim = head.simulate(torch.randn(2, 16), torch.randn(2, 16))
    assert sim["exit_pred"].dtype == torch.long
    assert sim["action_pred"].dtype == torch.long
    assert sim["exit_pred"].shape == (2,)


def test_total_on_zero_inputs():
    head = StateTransitionHead(StateTransitionConfig(dim=16, hidden=32, n_actions=8))
    out = head(torch.zeros(1, 16), torch.zeros(1, 16))
    assert torch.isfinite(out["delta"]).all()
    assert torch.isfinite(out["exit_logit"]).all()
    assert torch.isfinite(out["action_logits"]).all()
