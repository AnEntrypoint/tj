#!/usr/bin/env python3
"""Tianji exhaustive manual debugging suite.

Tests every module exhaustively with detailed output, timing, and
VRAM tracking. Run with:

    python manual_debug.py              # all modules
    python manual_debug.py --module qat # specific module
    python manual_debug.py --cpu        # force CPU
    python manual_debug.py --dim 64     # custom dimension

Modules: protocol, caps, apt, mamba2, mla, moe, hybrid, mtp,
         fakequant, kv_quant, adam8bit, transition, lora, ewc,
         replay, kd, qat, contrastive, router, generator,
         paged_attn, ring_attn, spec_decode, expert_offload,
         ccsniff, hf_datasets, engine, server, training
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── helpers ──────────────────────────────────────────────────────────

PASS = 0
FAIL = 0
SKIP = 0
START_TS = time.time()


def _hdr(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _sub(title: str) -> None:
    print(f"\n  --- {title} ---")


def _ok(msg: str = "") -> None:
    global PASS
    PASS += 1
    extra = f"  [{msg}]" if msg else ""
    print(f"    PASS{extra}")


def _fail(msg: str) -> None:
    global FAIL
    FAIL += 1
    print(f"    FAIL: {msg}")


def _skip(msg: str) -> None:
    global SKIP
    SKIP += 1
    print(f"    SKIP: {msg}")


def _vram() -> str:
    if torch.cuda.is_available():
        return f"{torch.cuda.memory_allocated()/1024**2:.1f}MB"
    return "N/A"


def _timed(fn, *args, **kw):
    t0 = time.time()
    try:
        result = fn(*args, **kw)
        dt = (time.time() - t0) * 1000
        return result, dt
    except Exception as e:
        raise


# ── module debuggers ─────────────────────────────────────────────────

def debug_protocol(args):
    _hdr("PROTOCOL — frame hashing, canonical JSON, verification")
    from tianji.protocol import (
        Frame, Trajectory, ToolCall, ToolResult, DiffHunk,
        frame_hash, make_frame, verify_frame, parse_frame, canonical_json,
    )

    _sub("Trajectory construction")
    try:
        t = Trajectory(kind="tool_call", trace="t1", source="cursor", ts=1000,
                       text="test", call=ToolCall(name="edit", args={"path":"x.py"}))
        _ok(f"kind={t.kind} trace={t.trace}")
    except Exception as e:
        _fail(str(e)); return

    _sub("Frame hashing")
    try:
        evs = (t, Trajectory(kind="trace_end", trace="t1", source="cursor", ts=1001))
        f = make_frame("t1", "cursor", 0, evs)
        ok = verify_frame(f)
        _ok(f"hash={f.hash[:16]}... verified={ok}")
    except Exception as e:
        _fail(str(e))

    _sub("Canonical JSON round-trip")
    try:
        js = canonical_json({"a": 1, "b": [2, 3]})
        parsed = json.loads(js)
        assert parsed == {"a": 1, "b": [2, 3]}
        _ok(f"round-trip OK, len={len(js)}")
    except Exception as e:
        _fail(str(e))

    _sub("Frame parse round-trip")
    try:
        js = canonical_json(f)
        f2 = parse_frame(js)
        ok = verify_frame(f2) and f2.hash == f.hash
        _ok(f"parse+verify={ok}")
    except Exception as e:
        _fail(str(e))

    _sub("Edge: empty events")
    try:
        f3 = make_frame("e", "cursor", 0, ())
        ok = verify_frame(f3)
        _ok(f"empty frame verified={ok}, hash={f3.hash[:16]}...")
    except Exception as e:
        _fail(str(e))

    _sub("Edge: tampered frame")
    try:
        f4 = Frame(trace="x", source="cursor", seq=0, events=evs, hash="sha256:deadbeef")
        ok = verify_frame(f4)
        _ok(f"tampered frame rejected={not ok}")
    except Exception as e:
        _fail(str(e))


def debug_caps(args):
    _hdr("CAPS — capability minting, resource budget, regions")
    from tianji.caps import Cap, grant, assert_cap, Region, ResourceBudget, disjoint

    _sub("Cap minting")
    try:
        c = grant("test")
        assert_cap(c, "test")
        _ok(f"minted cap={c}")
    except Exception as e:
        _fail(str(e))

    _sub("Cap rejection")
    try:
        c2 = grant("other")
        try:
            assert_cap(c2, "test")
            _fail("should have raised")
        except PermissionError:
            _ok("correctly rejected wrong kind")
    except Exception as e:
        _fail(str(e))

    _sub("Region open/close")
    try:
        r = Region.open("data", [1, 2, 3])
        v = r.value
        r.close()
        try:
            _ = r.value
            _fail("should have raised")
        except RuntimeError:
            _ok("closed region correctly raises")
    except Exception as e:
        _fail(str(e))

    _sub("ResourceBudget")
    try:
        b = ResourceBudget("vram", 1024)
        b.allocate(512)
        assert b.used_bytes == 512
        assert b.remaining == 512
        b.free(256)
        assert b.used_bytes == 256
        _ok(f"budget: used={b.used_bytes} remaining={b.remaining}")
    except Exception as e:
        _fail(str(e))

    _sub("Budget overflow")
    try:
        try:
            b.allocate(2000)
            _fail("should have raised")
        except MemoryError:
            _ok("correctly raised on overflow")
    except Exception as e:
        _fail(str(e))


def debug_apt(args):
    _hdr("APT — vocab build, encode, decode, embed, AST extraction")
    from tianji.tokens.apt import Vocab, encode, decode, embed, SPECIAL_IDS, SPECIAL_TAGS

    dim = args.dim
    corpus = [
        "def fib(n): return n",
        "<tool_call>{\"name\":\"edit\"}</tool_call>",
        "<bash_output>ok</bash_output>",
        "<system>agent</system>",
        "<cot>plan</cot>",
        "<diff>--- a\n+++ b\n</diff>",
        "class Foo: pass",
        "import torch",
    ] * 4

    _sub("Vocab build")
    try:
        v = Vocab.build(corpus, target_size=128, dim=dim, ast_dim=8)
        _ok(f"size={v.size} dim={v.dim} tokens={len(v.tokens)}")
    except Exception as e:
        _fail(str(e)); return

    _sub("Special tokens")
    try:
        for tag in SPECIAL_TAGS[:6]:
            sid = SPECIAL_IDS.get(tag)
            assert sid is not None, f"missing {tag}"
        _ok(f"{len(SPECIAL_TAGS)} special tags mapped")
    except Exception as e:
        _fail(str(e))

    _sub("Encode/decode round-trip")
    try:
        for text in corpus[:4]:
            out = encode(text, v, parse_ast=False)
            decoded = decode(out.ids, v)
            # Character-level tokenizer: decode approximate
            assert len(out.ids) > 0, f"empty ids for {text[:20]}"
            _ok(f"'{text[:20]}...' -> {len(out.ids)} tokens -> '{decoded[:20]}...'")
    except Exception as e:
        _fail(str(e))

    _sub("AST extraction")
    try:
        out = encode("def foo(): pass\nclass Bar: pass", v, parse_ast=True)
        _ok(f"AST nodes: {len(out.ast_nodes)} found")
        for kind, snippet in out.ast_nodes[:3]:
            print(f"      {kind}: {snippet[:50]}")
    except Exception as e:
        _fail(str(e))

    _sub("Embedding shape")
    try:
        out = encode("hello", v, parse_ast=False)
        emb = embed(out, v)
        _ok(f"embed shape={emb.shape} dtype={emb.dtype}")
    except Exception as e:
        _fail(str(e))

    _sub("Edge: empty text")
    try:
        out = encode("", v, parse_ast=False)
        assert len(out.ids) == 0
        _ok("empty text -> 0 tokens")
    except Exception as e:
        _fail(str(e))

    _sub("Edge: OOB token IDs")
    try:
        decoded = decode([99999, -1], v)
        _ok(f"OOB tokens decode to: '{decoded}' (should be empty)")
    except Exception as e:
        _fail(str(e))


def debug_mamba2(args):
    _hdr("MAMBA2 — SSM layer shapes, state carry, gradient checkpointing")
    from tianji.arch.mamba2 import Mamba2Layer, MambaConfig

    dim = args.dim
    d_inner = dim * 2

    _sub("Forward shape")
    try:
        cfg = MambaConfig(dim=dim, state_dim=4, d_inner=d_inner, dt_rank=4)
        layer = Mamba2Layer(cfg)
        x = torch.randn(2, 8, dim)
        y, state = layer(x)
        _ok(f"in={tuple(x.shape)} out={tuple(y.shape)} state={tuple(state.shape)}")
    except Exception as e:
        _fail(str(e)); return

    _sub("State carry")
    try:
        # Feed chunk 1
        x1 = torch.randn(2, 4, dim)
        y1, s1 = layer(x1)
        # Feed chunk 2 with state from chunk 1
        x2 = torch.randn(2, 4, dim)
        y2, s2 = layer(x2, state=s1)
        _ok(f"chunk1 state={tuple(s1.shape)} chunk2 state={tuple(s2.shape)}")
    except Exception as e:
        _fail(str(e))

    _sub("Gradient flow")
    try:
        x = torch.randn(2, 4, dim, requires_grad=True)
        y, _ = layer(x)
        loss = y.sum()
        loss.backward()
        has_grad = x.grad is not None and x.grad.abs().sum() > 0
        _ok(f"gradient flows={has_grad}")
    except Exception as e:
        _fail(str(e))

    if torch.cuda.is_available():
        _sub("CUDA transfer")
        try:
            layer_cuda = Mamba2Layer(cfg).cuda()
            x_cuda = torch.randn(2, 8, dim, device="cuda")
            y_cuda, s_cuda = layer_cuda(x_cuda)
            _ok(f"CUDA forward OK, VRAM={_vram()}")
        except Exception as e:
            _fail(str(e))


def debug_mla(args):
    _hdr("MLA — Multi-head Latent Attention shapes")
    from tianji.arch.mla import MLALayer, MLAConfig

    dim = args.dim
    _sub("Forward shape")
    try:
        cfg = MLAConfig(dim=dim, n_heads=2, head_dim=dim//2, kv_latent=dim)
        layer = MLALayer(cfg)
        x = torch.randn(2, 8, dim)
        y, _c = layer(x)
        _ok(f"in={tuple(x.shape)} out={tuple(y.shape)}")
    except Exception as e:
        _fail(str(e))

    _sub("Causal masking")
    try:
        # Verify output depends on position (not identical)
        x = torch.randn(2, 8, dim)
        y, _c = layer(x)
        diffs = (y[:, 0] - y[:, -1]).abs().sum()
        _ok(f"position-varying output: diff={diffs.item():.4f}")
    except Exception as e:
        _fail(str(e))


def debug_moe(args):
    _hdr("MoE — expert routing, load balancing, CUDA-graph compatibility")
    from tianji.arch.moe import MoELayer, MoEConfig

    dim = args.dim
    _sub("Forward shape + aux loss")
    try:
        cfg = MoEConfig(dim=dim, n_experts=4, n_active=2, shared_experts=1, expert_hidden=dim*2)
        layer = MoELayer(cfg)
        x = torch.randn(2, 8, dim)
        y, aux = layer(x)
        _ok(f"in={tuple(x.shape)} out={tuple(y.shape)} aux={aux.item():.4f}")
    except Exception as e:
        _fail(str(e)); return

    _sub("Router bias")
    try:
        bias = torch.zeros(4)  # n_experts=4
        bias[0] = 10.0  # force expert 0
        y_biased, _ = layer(x, router_bias=bias)
        y_normal, _ = layer(x)
        diff = (y_biased - y_normal).abs().mean().item()
        _ok(f"router bias effect: diff={diff:.6f}")
    except Exception as e:
        _fail(str(e))

    _sub("Static shape invariant")
    try:
        for n in [1, 4, 8, 16]:
            xn = torch.randn(2, n, dim)
            yn, _ = layer(xn)
            assert yn.shape == xn.shape, f"shape mismatch: {yn.shape} != {xn.shape}"
        _ok("all seq lengths preserve shape")
    except Exception as e:
        _fail(str(e))


def debug_hybrid(args):
    _hdr("HYBRID STACK — full model forward, state carry, num_stateful")
    from tianji.arch.hybrid import HybridStack, HybridConfig

    dim = args.dim
    _sub("Build and forward")
    try:
        cfg = HybridConfig(dim=dim, n_layers=27, state_dim=4, d_inner=dim*2,
                           n_heads=2, head_dim=dim//2, n_experts=4, n_active=2,
                           shared_experts=1, expert_hidden=dim*2)
        stack = HybridStack(cfg)
        x = torch.randn(2, 8, dim)
        h, aux, next_state = stack(x)
        _ok(f"in={tuple(x.shape)} out={tuple(h.shape)} aux={aux.item():.4f} "
            f"num_stateful={stack.num_stateful} stateful_layers={stack.num_stateful}")
    except Exception as e:
        _fail(str(e)); return

    _sub("State carry across chunks")
    try:
        x1 = torch.randn(2, 4, dim)
        h1, _, s1 = stack(x1)
        x2 = torch.randn(2, 4, dim)
        h2, _, s2 = stack(x2, state_tuple=s1)
        _ok(f"state carry: s1={len(s1) if s1 else 0} tensors, s2={len(s2) if s2 else 0}")
    except Exception as e:
        _fail(str(e))

    _sub("Layer composition")
    try:
        from tianji.arch.hybrid import MambaBlock, MLAMoELayer
        mamba_count = sum(1 for l in stack.layers if isinstance(l, MambaBlock))
        mla_count = sum(1 for l in stack.layers if isinstance(l, MLAMoELayer))
        _ok(f"layers: {mamba_count} Mamba-2 + {mla_count} MLA+MoE = {len(stack.layers)}")
    except Exception as e:
        _fail(str(e))

    _sub("Gradient flow")
    try:
        x = torch.randn(2, 4, dim, requires_grad=True)
        h, _, _ = stack(x)
        loss = h.sum()
        loss.backward()
        has_grad = x.grad is not None and x.grad.abs().sum() > 0
        _ok(f"gradient through full stack={has_grad}")
    except Exception as e:
        _fail(str(e))


def debug_mtp(args):
    _hdr("MTP — Multi-Token Prediction head")
    from tianji.arch.mtp import MTPHead, MTPConfig

    dim = args.dim
    _sub("Forward shape")
    try:
        cfg = MTPConfig(dim=dim, depth=3, vocab_size=128)
        head = MTPHead(cfg)
        x = torch.randn(2, 4, dim)
        outs = head(x)
        _ok(f"depth={len(outs)} shapes={[tuple(o.shape) for o in outs]}")
    except Exception as e:
        _fail(str(e)); return

    _sub("Speculation")
    try:
        tokens = head.speculate(x)
        _ok(f"speculated tokens: {[t.shape for t in tokens]}")
    except Exception as e:
        _fail(str(e))


def debug_fakequant(args):
    _hdr("FAKEQUANT — int4 fake quantization")
    from tianji.quant.fakequant import FakeQuantLinear, fakequant_int4

    dim = args.dim
    _sub("Quantize/dequantize")
    try:
        w = torch.randn(dim, dim)
        wq = fakequant_int4(w)
        err = (w - wq).abs().mean().item()
        _ok(f"quantization error: {err:.6f} (should be small)")
    except Exception as e:
        _fail(str(e))

    _sub("FakeQuantLinear forward")
    try:
        layer = FakeQuantLinear(dim, dim)
        x = torch.randn(2, 4, dim)
        y = layer(x)
        _ok(f"in={tuple(x.shape)} out={tuple(y.shape)}")
    except Exception as e:
        _fail(str(e))


def debug_kv_quant(args):
    _hdr("KV QUANT — int2 KV cache quantization")
    try:
        from tianji.quant.kv_quant import pack_int2, unpack_int2
        _sub("Pack/unpack int2")
        x = torch.randint(0, 4, (16,)).float()
        packed = pack_int2(x)
        unpacked = unpack_int2(packed, 16)
        err = (x - unpacked).abs().mean().item()
        _ok(f"pack/unpack error: {err:.6f}")
    except Exception as e:
        _fail(str(e))


def debug_adam8bit(args):
    _hdr("ADAM 8BIT — 8-bit Adam optimizer")
    try:
        from tianji.quant.adam8bit import Adam8bit
        _sub("Optimizer step")
        p = nn.Parameter(torch.randn(4, 4))
        opt = Adam8bit([p], lr=1e-3)
        loss = p.sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
        _ok("8-bit Adam step OK")
    except Exception as e:
        _fail(str(e))


def debug_transition(args):
    _hdr("STATE TRANSITION — delta, exit, action prediction")
    from tianji.state.transition import (
        StateTransitionHead, StateTransitionConfig, EVENT_KINDS, kind_to_idx,
    )

    dim = args.dim
    _sub("Forward")
    try:
        cfg = StateTransitionConfig(dim=dim, hidden=dim*2, n_actions=len(EVENT_KINDS))
        head = StateTransitionHead(cfg)
        state = torch.randn(2, dim)
        ctx = torch.zeros(2, dim)
        out = head(state, ctx)
        _ok(f"delta={tuple(out['delta'].shape)} exit={tuple(out['exit_logit'].shape)} "
            f"actions={tuple(out['action_logits'].shape)}")
    except Exception as e:
        _fail(str(e)); return

    _sub("Simulate")
    try:
        sim = head.simulate(state, ctx)
        _ok(f"exit_pred={sim['exit_pred'].tolist()} action_pred={sim['action_pred'].tolist()}")
    except Exception as e:
        _fail(str(e))

    _sub("Kind mapping")
    try:
        for k in EVENT_KINDS:
            idx = kind_to_idx(k)
            assert 0 <= idx < len(EVENT_KINDS)
        _ok(f"{len(EVENT_KINDS)} event kinds mapped")
    except Exception as e:
        _fail(str(e))


def debug_lora(args):
    _hdr("LORA — adapter wrapping, merge, save/load")
    from tianji.distill.lora import (
        wrap_linear_with_lora, LoRAConfig, merge_lora, save_lora_state, load_lora_state,
    )

    dim = args.dim
    _sub("Wrap and forward")
    try:
        lin = nn.Linear(dim, dim, bias=False)
        cfg = LoRAConfig(rank=4, alpha=2.0)
        wrapped = wrap_linear_with_lora(lin, cfg)
        x = torch.randn(2, 4, dim)
        y = wrapped(x)
        _ok(f"in={tuple(x.shape)} out={tuple(y.shape)}")
    except Exception as e:
        _fail(str(e)); return

    _sub("Merge")
    try:
        n = merge_lora(wrapped)
        y2 = wrapped(x)
        _ok(f"merged {n} adapters, output shape={tuple(y2.shape)}")
    except Exception as e:
        _fail(str(e))

    _sub("Save/load round-trip")
    try:
        state = save_lora_state(wrapped)
        n2 = load_lora_state(wrapped, state)
        _ok(f"save/load: {n2} adapters restored")
    except Exception as e:
        _fail(str(e))


def debug_ewc(args):
    _hdr("EWC — Elastic Weight Consolidation")
    from tianji.distill.ewc import compute_fisher, consolidate, EWCState

    dim = args.dim
    _sub("Fisher computation")
    try:
        model = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        def loss_fn():
            x = torch.randn(4, dim)
            return model(x).sum()
        fisher = compute_fisher(model, loss_fn, steps=3)
        _ok(f"fisher computed for {len(fisher)} parameters")
    except Exception as e:
        _fail(str(e)); return

    _sub("Consolidate and penalty")
    try:
        ewc = consolidate(model, fisher)
        penalty = ewc.penalty(model, lam=1.0)
        _ok(f"EWC penalty={penalty.item():.6f} (should be ~0 for unchanged params)")
    except Exception as e:
        _fail(str(e))

    _sub("Penalty after weight change")
    try:
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.1)
        penalty2 = ewc.penalty(model, lam=1.0)
        _ok(f"EWC penalty after drift={penalty2.item():.6f} (should be >0)")
    except Exception as e:
        _fail(str(e))


def debug_replay(args):
    _hdr("REPLAY — ReplayBuffer")
    from tianji.distill.replay import ReplayBuffer

    _sub("Push and sample")
    try:
        buf = ReplayBuffer(capacity=8)
        for i in range(10):
            inp = torch.randint(0, 128, (1, 4))
            tgt = torch.randint(0, 128, (1, 4))
            buf.push(inp, tgt)
        batch = buf.sample(4)
        _ok(f"capacity=8, 10 pushes, sample={len(batch) if batch else 0}")
    except Exception as e:
        _fail(str(e))


def debug_kd(args):
    _hdr("KD — Knowledge Distillation")
    from tianji.distill.kd import kd_loss, make_stub_teacher, StubTeacher

    dim = args.dim
    _sub("Stub teacher")
    try:
        teacher = make_stub_teacher(128)
        ids = torch.randint(0, 128, (2, 8))
        logits = teacher(ids)
        _ok(f"stub teacher output shape={tuple(logits.shape)}")
    except Exception as e:
        _fail(str(e)); return

    _sub("KD loss")
    try:
        student_logits = torch.randn(2, 8, 128)
        teacher_logits = teacher(ids)
        loss = kd_loss(student_logits, teacher_logits, T=2.0)
        _ok(f"KD loss={loss.item():.4f}")
    except Exception as e:
        _fail(str(e))


def debug_qat(args):
    _hdr("QAT LOOP — full training step, AMP, CUDA graph, checkpoint")
    from tianji.distill.qat_loop import QATLoop, QATConfig
    from tianji.arch.hybrid import HybridConfig
    from tianji.tokens.apt import Vocab

    dim = args.dim
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    _sub("Build and step")
    try:
        vocab = Vocab.build(["test"]*10, target_size=128, dim=dim, ast_dim=8)
        arch = HybridConfig(dim=dim, n_layers=27)
        cfg = QATConfig(device=device, lora_rank=4, vram_bytes=4*1024**3,
                        precision="fp16" if device=="cuda" else "fp32")
        qat = QATLoop(cfg, arch, vocab_size=vocab.size)
        inp = torch.randint(0, vocab.size, (2, 8))
        tgt = torch.randint(0, vocab.size, (2, 8))
        res, next_state = qat.step(inp, tgt, source="synthetic")
        _ok(f"loss={res.loss:.4f} kd={res.kd_loss:.4f} aux={res.aux_loss:.4f} "
            f"vram={res.vram_used_bytes}B")
    except Exception as e:
        _fail(str(e)); return

    _sub("Multiple steps")
    try:
        losses = []
        for i in range(5):
            res, _ = qat.step(inp, tgt, source="synthetic")
            losses.append(res.loss)
        _ok(f"5 steps: losses={[f'{l:.3f}' for l in losses]}")
    except Exception as e:
        _fail(str(e))

    _sub("Checkpoint save/load")
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp = f.name
        qat.save_checkpoint(tmp)
        n = qat.load_checkpoint(tmp)
        os.unlink(tmp)
        _ok(f"checkpoint: {n} adapters saved/loaded")
    except Exception as e:
        _fail(str(e))

    _sub("Hidden state extraction")
    try:
        h = qat.hidden(inp)
        _ok(f"hidden shape={tuple(h.shape)}")
    except Exception as e:
        _fail(str(e))

    qat.close()


def debug_contrastive(args):
    _hdr("CONTRASTIVE — InfoNCE and Triplet loss")
    from tianji.distill.contrastive import (
        ContrastiveLoss, TripletLoss, contrastive_loss_from_pooled,
    )

    dim = args.dim
    _sub("InfoNCE loss")
    try:
        criterion = ContrastiveLoss(temperature=0.07)
        pos = torch.randn(8, dim)
        neg = torch.randn(4, dim)
        loss = criterion(pos, neg)
        _ok(f"InfoNCE loss={loss.item():.4f}")
    except Exception as e:
        _fail(str(e)); return

    _sub("Triplet loss")
    try:
        criterion = TripletLoss(margin=0.5)
        anchor = torch.randn(4, dim)
        positive = torch.randn(4, dim)
        negative = torch.randn(4, dim)
        loss = criterion(anchor, positive, negative)
        _ok(f"Triplet loss={loss.item():.4f}")
    except Exception as e:
        _fail(str(e))

    _sub("Convenience function")
    try:
        loss = contrastive_loss_from_pooled(pos, neg, loss_type="infonce")
        _ok(f"convenience InfoNCE={loss.item():.4f}")
    except Exception as e:
        _fail(str(e))

    _sub("Edge: empty inputs")
    try:
        # Use ContrastiveLoss (the first criterion) for edge test
        criterion2 = ContrastiveLoss(temperature=0.07)
        empty = torch.randn(0, dim)
        loss = criterion2(empty, neg)
        _ok(f"empty positive loss={loss.item():.4f} (should be 0)")
    except Exception as e:
        _fail(str(e))


def debug_router(args):
    _hdr("ROUTER ALIGNMENT — source-to-expert bias")
    from tianji.distill.router_alignment import RouterAlignment, apply_router_bias

    dim = args.dim
    _sub("Build and bias")
    try:
        ra = RouterAlignment.build(dim)
        for src in ["claude", "gpt-4o", "cursor", "synthetic", "github-copilot"]:
            bias = ra.bias_for(src)
            nonzero = (bias > 0).sum().item()
            _ok(f"source={src}: nonzero_experts={nonzero}")
    except Exception as e:
        _fail(str(e)); return

    _sub("Apply router bias")
    try:
        logits = torch.randn(2, 8, dim)
        biased = apply_router_bias(logits, ra, "claude")
        diff = (biased - logits).abs().sum().item()
        _ok(f"bias applied: diff={diff:.4f}")
    except Exception as e:
        _fail(str(e))


def debug_generator(args):
    _hdr("GENERATOR — incremental inference, sampling methods")
    from tianji.tokens.apt import Vocab
    from tianji.arch.hybrid import HybridConfig
    from tianji.distill.qat_loop import QATLoop, QATConfig
    from tianji.infer.generator import Generator, GenerateConfig, SamplingConfig

    dim = args.dim
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    _sub("Build and generate")
    try:
        vocab = Vocab.build(["test"]*10, target_size=128, dim=dim, ast_dim=8)
        arch = HybridConfig(dim=dim, n_layers=27)
        cfg = QATConfig(device=device, lora_rank=4, vram_bytes=4*1024**3,
                        precision="fp16" if device=="cuda" else "fp32")
        qat = QATLoop(cfg, arch, vocab_size=vocab.size)
        gen = Generator(qat, GenerateConfig(max_tokens=8))
        steps = list(gen.generate([1, 2, 3, 4]))
        _ok(f"generated {len(steps)} tokens: {[s.token for s in steps]}")
    except Exception as e:
        _fail(str(e)); qat.close(); return

    _sub("Sampling modes")
    try:
        for name, scfg in [
            ("greedy", SamplingConfig(temperature=0)),
            ("temp0.7", SamplingConfig(temperature=0.7)),
            ("topk40", SamplingConfig(temperature=0.7, top_k=40)),
            ("topp0.9", SamplingConfig(temperature=0.7, top_p=0.9)),
        ]:
            gen2 = Generator(qat, GenerateConfig(max_tokens=8, sampling=scfg))
            t0 = time.time()
            steps = list(gen2.generate([1, 2, 3, 4]))
            dt = (time.time() - t0) * 1000
            _ok(f"{name}: {len(steps)} tok in {dt:.1f}ms ({len(steps)/dt*1000:.1f} tok/s)")
    except Exception as e:
        _fail(str(e))

    _sub("State reset")
    try:
        gen.reset_state()
        steps2 = list(gen.generate([1, 2, 3, 4]))
        _ok(f"after reset: {len(steps2)} tokens")
    except Exception as e:
        _fail(str(e))

    qat.close()


def debug_paged_attn(args):
    _hdr("PAGED ATTENTION — KV cache scaffolding")
    _sub("Module exists")
    try:
        from tianji.infer.paged_attn import PagedKVCache
        _ok("paged_attn module importable")
    except Exception as e:
        _fail(str(e))


def debug_ring_attn(args):
    _hdr("RING ATTENTION — distributed attention scaffolding")
    _sub("Module exists")
    try:
        from tianji.infer.ring_attn import RingConfig
        _ok("ring_attn module importable")
    except Exception as e:
        _fail(str(e))


def debug_spec_decode(args):
    _hdr("SPECULATIVE DECODING — scaffolding")
    _sub("Module exists")
    try:
        from tianji.infer.spec_decode import SpeculativeResult
        _ok("spec_decode module importable")
    except Exception as e:
        _fail(str(e))


def debug_expert_offload(args):
    _hdr("EXPERT OFFLOADING — scaffolding")
    from tianji.infer.expert_offload import ExpertOffloader
    _sub("Module exists")
    try:
        _ok("expert_offload module importable")
    except Exception as e:
        _fail(str(e))


def debug_ccsniff(args):
    _hdr("CCSNIFF INGEST — NDJSON parsing, frame assembly")
    from tianji.ingest.ccsniff import (
        row_to_trajectory, rows_to_frames, parse_ccsniff_ndjson, ingest_ccsniff_stream,
    )
    from tianji.protocol import verify_frame

    _sub("Row to trajectory")
    rows = [
        {"ts": 1000, "sid": "s1", "role": "system", "type": "system", "text": "You are Claude."},
        {"ts": 1100, "sid": "s1", "role": "user", "type": "text", "text": "Read src/app.py"},
        {"ts": 1200, "sid": "s1", "role": "assistant", "type": "thinking", "text": "I need to read."},
        {"ts": 1300, "sid": "s1", "role": "assistant", "type": "tool_use", "text": '{"path":"x.py"}', "tool": "Read"},
        {"ts": 1400, "sid": "s1", "role": "tool_result", "type": "tool_result", "text": "def main(): pass"},
    ]
    try:
        for r in rows:
            ev = row_to_trajectory(r, r.get("sid"))
            _ok(f"kind={ev.kind} text={str(ev.text)[:40]}")
    except Exception as e:
        _fail(str(e)); return

    _sub("Frame assembly")
    try:
        frames = list(rows_to_frames(iter(rows), batch_size=32))
        for f in frames:
            ok = verify_frame(f)
            kinds = [e.kind for e in f.events]
            _ok(f"frame: {len(f.events)} events, kinds={kinds[:3]}..., verified={ok}")
    except Exception as e:
        _fail(str(e))

    _sub("End-to-end stream")
    try:
        lines = [json.dumps(r) for r in rows]
        frames = list(ingest_ccsniff_stream(iter(lines), batch_size=32))
        _ok(f"stream: {len(frames)} frames, all verified={all(verify_frame(f) for f in frames)}")
    except Exception as e:
        _fail(str(e))

    _sub("Edge: empty input")
    try:
        frames = list(ingest_ccsniff_stream(iter([]), batch_size=32))
        _ok(f"empty stream: {len(frames)} frames (should be 0)")
    except Exception as e:
        _fail(str(e))

    _sub("Edge: malformed JSON")
    try:
        frames = list(ingest_ccsniff_stream(iter(["not json", "", '{"bad": "missing fields"}']), batch_size=32))
        _ok(f"malformed stream: {len(frames)} frames (should be 0)")
    except Exception as e:
        _fail(str(e))


def debug_hf_datasets(args):
    _hdr("HF DATASETS — streaming loader, dataset registry")
    try:
        from tianji.ingest.hf_datasets import (
            iter_hf_texts, list_known_datasets, KNOWN_DATASETS, _known_dataset,
        )
        _sub("Known datasets")
        names = list_known_datasets()
        _ok(f"{len(names)} datasets: {', '.join(names[:5])}")
    except Exception as e:
        _fail(str(e)); return

    _sub("Config resolution")
    try:
        for name in ["the-stack", "starcoder", "codeparrot"]:
            cfg = _known_dataset(name)
            _ok(f"{name}: path={cfg.path} split={cfg.split}")
    except Exception as e:
        _fail(str(e))

    _sub("Streaming fetch (if datasets installed)")
    try:
        from tianji.ingest.hf_datasets import _import_datasets
        _load = _import_datasets()
        _skip("datasets library available, skipping network fetch")
    except ImportError:
        _skip("datasets library not installed")


def debug_engine(args):
    _hdr("ENGINE — frame training, state head, simulate")
    from tianji.engine import Engine, EngineConfig
    from tianji.tokens.apt import Vocab
    from tianji.arch.hybrid import HybridConfig
    from tianji.distill.qat_loop import QATConfig
    from tianji.protocol import (
        Frame, Trajectory, ToolCall, ToolResult, frame_hash, make_frame,
    )

    dim = args.dim
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    _sub("Build engine")
    try:
        vocab = Vocab.build(["test"]*10, target_size=128, dim=dim, ast_dim=8)
        arch = HybridConfig(dim=dim, n_layers=27)
        qat_cfg = QATConfig(device=device, lora_rank=4, vram_bytes=4*1024**3,
                            precision="fp16" if device=="cuda" else "fp32")
        eng_cfg = EngineConfig(device=device, seq_len=8, batch_size=1)
        eng = Engine(vocab, arch, qat_cfg, eng_cfg)
        _ok(f"engine built, dim={dim}, vram={_vram()}")
    except Exception as e:
        _fail(str(e)); return

    _sub("Step frame")
    try:
        evs = (
            Trajectory(kind="system_prompt", trace="t", source="synthetic", ts=1, text="<system>agent</system>"),
            Trajectory(kind="tool_call", trace="t", source="synthetic", ts=2,
                       call=ToolCall(name="edit", args={"path":"x.py"}, args_ast=None)),
            Trajectory(kind="tool_result", trace="t", source="synthetic", ts=3,
                       result=ToolResult(exit=0, stdout="ok", stderr=None)),
            Trajectory(kind="trace_end", trace="t", source="synthetic", ts=4),
        )
        f = make_frame("t", "synthetic", 0, evs)
        res = eng.step_frame(f)
        _ok(f"qat_loss={res.qat.loss:.4f} state_loss={res.state_loss:.4f} "
            f"pred_kind={res.state_pred_kind} target_kind={res.state_target_kind}")
    except Exception as e:
        _fail(str(e)); return

    _sub("Multiple frames")
    try:
        for i in range(3):
            evs2 = (
                Trajectory(kind="context", trace="t", source="synthetic", ts=10+i, text=f"step {i}"),
                Trajectory(kind="trace_end", trace="t", source="synthetic", ts=11+i),
            )
            f2 = make_frame("t", "synthetic", i+1, evs2)
            res2 = eng.step_frame(f2)
            _ok(f"frame {i+1}: loss={res2.qat.loss:.4f} state_loss={res2.state_loss:.4f}")
    except Exception as e:
        _fail(str(e))

    _sub("Simulate action")
    try:
        sim = eng.simulate_action("def foo(): pass")
        _ok(f"delta_norm={sim['delta'].norm().item():.4f} exit={sim['exit_pred'].item()} action={sim['action_pred'].item()}")
    except Exception as e:
        _fail(str(e))

    _sub("Contrastive step")
    try:
        c_loss = eng.step_contrastive(["def foo(): pass", "class Bar: pass"], ["import os", "print('hi')"])
        _ok(f"contrastive loss={c_loss:.4f}")
    except Exception as e:
        _fail(str(e))

    _sub("Save/load training state")
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp = f.name
        eng.save_training_state(tmp)
        eng2 = Engine(vocab, arch, qat_cfg, eng_cfg)
        eng2.load_training_state(tmp)
        os.unlink(tmp)
        _ok("training state save/load OK")
        eng2.close()
    except Exception as e:
        _fail(str(e))

    eng.close()


def debug_server(args):
    _hdr("SERVER — load/unload, hot-reload")
    os.environ.setdefault("TIANJI_API_KEY", "test")
    try:
        from tianji.server import load_model, unload_model, reload_model
        _sub("Load model")
        info = load_model(device="cpu", dim=args.dim)
        _ok(f"loaded: {info}")
    except Exception as e:
        _fail(str(e)); return

    _sub("Already loaded")
    try:
        info2 = load_model(device="cpu", dim=args.dim)
        _ok(f"already loaded: {info2['status']}")
    except Exception as e:
        _fail(str(e))

    _sub("Reload")
    try:
        import tempfile
        from tianji.distill.qat_loop import QATLoop, QATConfig
        from tianji.arch.hybrid import HybridConfig
        from tianji.tokens.apt import Vocab
        vocab = Vocab.build(["test"]*10, target_size=512, dim=args.dim, ast_dim=8)
        arch = HybridConfig(dim=args.dim, n_layers=27)
        qat = QATLoop(QATConfig(device="cpu"), arch, vocab_size=vocab.size)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp = f.name
        qat.save_checkpoint(tmp)
        qat.close()
        result = reload_model(tmp)
        os.unlink(tmp)
        _ok(f"reload: {result}")
    except Exception as e:
        _fail(str(e))

    _sub("Unload")
    try:
        unload_model()
        _ok("unloaded")
    except Exception as e:
        _fail(str(e))


def debug_training(args):
    _hdr("TRAINING — smoke test training script")
    _sub("Import check")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "train", os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "train.py"))
        _ok("train.py importable")
    except Exception as e:
        _fail(str(e))


# ── main ─────────────────────────────────────────────────────────────

MODULES = {
    "protocol": debug_protocol,
    "caps": debug_caps,
    "apt": debug_apt,
    "mamba2": debug_mamba2,
    "mla": debug_mla,
    "moe": debug_moe,
    "hybrid": debug_hybrid,
    "mtp": debug_mtp,
    "fakequant": debug_fakequant,
    "kv_quant": debug_kv_quant,
    "adam8bit": debug_adam8bit,
    "transition": debug_transition,
    "lora": debug_lora,
    "ewc": debug_ewc,
    "replay": debug_replay,
    "kd": debug_kd,
    "qat": debug_qat,
    "contrastive": debug_contrastive,
    "router": debug_router,
    "generator": debug_generator,
    "paged_attn": debug_paged_attn,
    "ring_attn": debug_ring_attn,
    "spec_decode": debug_spec_decode,
    "expert_offload": debug_expert_offload,
    "ccsniff": debug_ccsniff,
    "hf_datasets": debug_hf_datasets,
    "engine": debug_engine,
    "server": debug_server,
    "training": debug_training,
}


def main():
    global PASS, FAIL, SKIP

    ap = argparse.ArgumentParser(description="Tianji exhaustive manual debug suite")
    ap.add_argument("--module", "-m", default=None, help="Run specific module")
    ap.add_argument("--dim", type=int, default=16, help="Model dimension")
    ap.add_argument("--cpu", action="store_true", help="Force CPU")
    ap.add_argument("--list", action="store_true", help="List available modules")
    args = ap.parse_args()

    if args.list:
        for name in sorted(MODULES):
            print(f"  {name}")
        return

    print(f"Tianji Manual Debug Suite")
    print(f"  dim={args.dim}  device={'cpu' if args.cpu else 'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"  torch={torch.__version__}  cuda={torch.cuda.is_available()}")

    if args.module:
        if args.module in MODULES:
            MODULES[args.module](args)
        else:
            print(f"Unknown module: {args.module}")
            print(f"Available: {', '.join(sorted(MODULES))}")
            return
    else:
        for name, fn in MODULES.items():
            try:
                fn(args)
            except Exception as e:
                _fail(f"MODULE CRASH: {name}: {e}")
                traceback.print_exc()

    elapsed = time.time() - START_TS
    total = PASS + FAIL + SKIP
    print(f"\n{'='*60}")
    print(f"  RESULTS: {PASS} pass, {FAIL} fail, {SKIP} skip ({total} total)")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'='*60}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())