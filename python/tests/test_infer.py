import torch
import pytest
from tianji.infer.paged_attn import PagedKVCache, BLOCK_TOKENS
from tianji.infer.ring_attn import RingConfig, ring_chunk, ring_attn_forward
from tianji.infer.spec_decode import speculative_step
from tianji.infer.expert_offload import ExpertOffloader
from tianji.infer.generator import Generator, GenerateConfig
from tianji.arch.hybrid import HybridStack, HybridConfig
from tianji.arch.mtp import MTPHead, MTPConfig
from tianji.distill.qat_loop import QATLoop, QATConfig


def test_paged_kv_alloc_free_roundtrip():
    c = PagedKVCache.open(8)
    s = c.alloc_sequence()
    c.extend(s, BLOCK_TOKENS * 3)
    assert len(c.sequences[s]) == 3
    c.free_sequence(s)
    assert len(c.free) == 8


def test_paged_kv_oom_when_exhausted():
    c = PagedKVCache.open(2)
    s = c.alloc_sequence()
    c.extend(s, BLOCK_TOKENS * 2)
    s2 = c.alloc_sequence()
    with pytest.raises(MemoryError):
        c.extend(s2, BLOCK_TOKENS)


def test_ring_chunk_overlap():
    cfg = RingConfig(ring_size=4, overlap=1)
    chunks = ring_chunk(torch.arange(10).view(1, 10, 1).float(), cfg)
    assert len(chunks) == 3
    assert chunks[0].shape == (1, 4, 1)
    assert chunks[1].shape == (1, 5, 1)
    assert chunks[2].shape == (1, 3, 1)


def test_speculative_step_returns_result():
    cfg = HybridConfig(dim=16, n_layers=3)
    model = HybridStack(cfg)
    embed = torch.nn.Embedding(32, 16)
    head = torch.nn.Linear(16, 32, bias=False)
    mtp = MTPHead(MTPConfig(dim=16, vocab_size=32, depth=2))
    ctx = torch.randint(0, 32, (2, 4))
    res = speculative_step(embed, model, head, mtp, ctx)
    assert res.total == 2
    assert 0 <= res.accepted <= 2
    assert res.next_token.shape == (2,)


def test_expert_offloader_fifo_eviction():
    off = ExpertOffloader(n_experts=4, n_resident=2)
    off.route(0); off.route(1)
    assert set(off.resident) == {0, 1}
    off.route(2)
    assert 0 not in off.resident


def test_generator_yields_tokens():
    qat_cfg = QATConfig(device="cpu", lora_rank=4, vram_bytes=4 * 1024 ** 3)
    qat = QATLoop(qat_cfg, HybridConfig(dim=16, n_layers=3), vocab_size=64)
    gen = Generator(qat, GenerateConfig(max_tokens=4, paged_kv_blocks=8))
    steps = list(gen.generate([1, 2, 3, 4]))
    assert len(steps) == 4
    for s in steps:
        assert 0 <= s.token < 64
    qat.close()
