"""Tianji CLI: demo, infer, ingest-ccsniff, checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from .tokens.apt import Vocab
from .arch.hybrid import HybridConfig
from .distill.qat_loop import QATConfig, QATLoop
from .caps import ResourceBudget


def _default_vocab():
    lines = [
        '<tool_call>{"name":"edit","args":{"path":"a.py"}}</tool_call>',
        "<bash_output>ok</bash_output>",
        "def fib(n): return n",
        "<system>agent</system>",
        "<cot>plan</cot>",
        "<diff>--- a\n+++ b\n</diff>",
    ] * 4
    return Vocab.build(lines, target_size=128, dim=16, ast_dim=8)


def _default_arch(dim=16, n_layers=27):
    return HybridConfig(dim=dim, n_layers=n_layers)


def _default_qat(device="cpu", dim=16):
    return QATConfig(device=device, lora_rank=4, vram_bytes=4 * 1024 ** 3)


def _vram_bytes_used(qat: QATLoop) -> int:
    total = 0
    for p in qat.model.parameters():
        total += p.numel() * p.element_size()
    return total


def cmd_demo(args):
    vocab = _default_vocab()
    arch = _default_arch()
    qat = QATLoop(_default_qat(), arch, vocab_size=vocab.size)
    print(f"[demo] built model with vocab_size={vocab.size}, layers={arch.n_layers}")
    ids = torch.randint(0, vocab.size, (2, 8))
    res = qat.step(ids, ids.roll(-1, dims=1), source="synthetic")
    print(f"[demo] step loss={res.loss:.4f} kd={res.kd_loss:.4f} vram={res.vram_used_bytes} bytes")
    qat.close()
    print("[demo] ok")


def cmd_infer(args):
    vocab = _default_vocab()
    arch = _default_arch(dim=args.dim, n_layers=args.layers)
    qat = QATLoop(_default_qat(), arch, vocab_size=vocab.size)
    from .infer.generator import Generator, GenerateConfig
    prompt = [1, 2, 3, 4]
    gen = Generator(qat, GenerateConfig(max_tokens=args.n, paged_kv_blocks=8))
    toks = [s.token for s in gen.generate(prompt)]
    print(f"[infer] generated {len(toks)} tokens: {toks}")
    qat.close()


def cmd_ingest_ccsniff(args):
    from .ingest.ccsniff import ingest_ccsniff_stream, verify_frame
    rows = []
    if args.jsonl:
        for line in Path(args.jsonl).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    else:
        rows = _sample_rows()
    frames = list(ingest_ccsniff_stream(iter(json.dumps(r) for r in rows), batch_size=args.batch))
    verified = sum(1 for f in frames if verify_frame(f))
    print(f"[ingest-ccsniff] produced {len(frames)} frames, {verified} verified")
    for f in frames:
        print(f"  frame trace={f.trace} events={len(f.events)} source={f.source}")


def _sample_rows():
    return [
        {"ts": 1000, "sid": "s1", "role": "system", "type": "system", "text": "You are Claude Code.", "tool": None, "isMeta": False, "isError": False},
        {"ts": 1200, "sid": "s1", "role": "user", "type": "text", "isMeta": False, "text": "Read src/app.py", "tool": None, "isError": False},
        {"ts": 1400, "sid": "s1", "role": "assistant", "type": "tool_use", "text": '{"file_path":"src/app.py"}', "tool": "Read", "isMeta": False, "isError": False},
        {"ts": 1500, "sid": "s1", "role": "tool_result", "type": "tool_result", "text": "def main(): pass", "tool": None, "isMeta": False, "isError": False},
        {"ts": 1700, "sid": "s1", "role": "result", "type": "result", "text": "", "tool": None, "isMeta": False, "isError": False, "duration": 500},
    ]


def cmd_checkpoint(args):
    vocab = _default_vocab()
    arch = _default_arch()
    qat = QATLoop(_default_qat(), arch, vocab_size=vocab.size)
    ids = torch.randint(0, vocab.size, (2, 8))
    qat.step(ids, ids.roll(-1, dims=1), source="synthetic")
    if args.action == "save":
        qat.save_checkpoint(args.path)
        print(f"[checkpoint] saved to {args.path}")
    elif args.action == "load":
        n = qat.load_checkpoint(args.path)
        print(f"[checkpoint] loaded {n} lora adapters from {args.path}")
    qat.close()


def main(argv=None):
    p = argparse.ArgumentParser(prog="tianji")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("demo")

    pi = sub.add_parser("infer")
    pi.add_argument("--prompt", default="def fib(n): return n")
    pi.add_argument("--n", type=int, default=8)
    pi.add_argument("--dim", type=int, default=16)
    pi.add_argument("--layers", type=int, default=27)

    pc = sub.add_parser("ingest-ccsniff")
    pc.add_argument("--jsonl", default=None)
    pc.add_argument("--batch", type=int, default=32)

    pck = sub.add_parser("checkpoint")
    pck.add_argument("action", choices=["save", "load"])
    pck.add_argument("path")

    args = p.parse_args(argv)
    dispatch = {
        "demo": cmd_demo,
        "infer": cmd_infer,
        "ingest-ccsniff": cmd_ingest_ccsniff,
        "checkpoint": cmd_checkpoint,
    }
    dispatch[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
