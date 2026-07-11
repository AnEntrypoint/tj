import torch
from tianji.arch.mamba2 import Mamba2Layer, MambaConfig
from tianji.arch.mla import MLALayer, MLAConfig
from tianji.arch.moe import MoELayer, MoEConfig
from tianji.arch.hybrid import HybridStack, HybridConfig, MambaBlock, MLAMoELayer
from tianji.arch.mtp import MTPHead, MTPConfig


def test_mamba2_forward_shape():
    cfg = MambaConfig(dim=16, state_dim=4, d_inner=32, dt_rank=4)
    y, state = Mamba2Layer(cfg)(torch.randn(2, 5, 16))
    assert y.shape == (2, 5, 16)
    assert state.shape == (2, 4, 32)


def test_mla_forward_shape():
    cfg = MLAConfig(dim=16, n_heads=2, head_dim=8, kv_latent=8)
    y, kv = MLALayer(cfg)(torch.randn(2, 4, 16))
    assert y.shape == (2, 4, 16)
    assert kv.shape == (2, 4, 8)


def test_moe_forward_shape_and_aux():
    cfg = MoEConfig(dim=16, n_experts=4, n_active=2, shared_experts=1, expert_hidden=32)
    y, aux = MoELayer(cfg)(torch.randn(2, 3, 16))
    assert y.shape == (2, 3, 16)
    assert torch.isfinite(aux)


def test_hybrid_27_layers():
    assert len(HybridStack(HybridConfig(dim=16, n_layers=27)).layers) == 27


def test_hybrid_segmentation():
    model = HybridStack(HybridConfig(dim=16, n_layers=27))
    assert sum(1 for l in model.layers if isinstance(l, MambaBlock)) == 18
    assert sum(1 for l in model.layers if isinstance(l, MLAMoELayer)) == 9


def test_hybrid_forward_shape():
    y, aux = HybridStack(HybridConfig(dim=16, n_layers=27))(torch.randn(1, 4, 16))
    assert y.shape == (1, 4, 16)
    assert torch.isfinite(aux)


def test_mtp_head_output_count():
    head = MTPHead(MTPConfig(dim=16, vocab_size=32, depth=2))
    outs = head(torch.randn(2, 5, 16))
    assert len(outs) == 2
    for o in outs:
        assert o.shape == (2, 5, 32)


def test_mtp_speculate_returns_k_tokens():
    head = MTPHead(MTPConfig(dim=16, vocab_size=32, depth=3))
    toks = head.speculate(torch.randn(2, 16))
    assert len(toks) == 3
    for t in toks:
        assert t.shape == (2,)
