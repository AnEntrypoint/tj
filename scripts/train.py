#!/usr/bin/env python3
"""Training driver: roll up Claude Code sessions via npx ccsniff, distill them
through Tianji's QAT loop, and keep npx ccwatch visible as the cost/quota monitor.

Usage:
  python scripts/train.py --steps 20 --batch 32
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from tianji import Engine, EngineConfig
from tianji.tokens.apt import Vocab
from tianji.arch.hybrid import HybridConfig
from tianji.distill.qat_loop import QATConfig
from tianji.ingest.ccsniff import ingest_ccsniff_stream
from tianji.protocol import verify_frame


def _collect_ccsniff_rows() -> list:
    r = subprocess.run(
        ["npx.cmd", "--yes", "ccsniff@latest", "--json", "--since", "1h", "--limit", "2000"],
        capture_output=True, shell=True,
    )
    if r.returncode != 0:
        print(f"[train] ccsniff collect failed: {r.stderr.decode('utf-8', 'replace').strip()}", file=sys.stderr)
        return []
    out = r.stdout.decode("utf-8", "replace") if r.stdout else ""
    return [ln for ln in out.splitlines() if ln.strip()]


def _train_on_rows(rows: list, batch: int) -> float:
    global _ENGINE
    frames = list(ingest_ccsniff_stream(iter(rows), batch_size=batch))
    total_loss = 0.0
    n = 0
    for f in frames:
        if not verify_frame(f):
            continue
        res = _ENGINE.step_frame(f)
        total_loss += res.qat.loss
        n += 1
    return total_loss / max(1, n)


_ENGINE = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--layers", type=int, default=27)
    ap.add_argument("--seq-len", type=int, default=8)
    args = ap.parse_args()

    vocab = Vocab.build(
        ['<tool_call>{"name":"edit"}</tool_call>', "<bash_output>ok</bash_output>",
         "def f(x): return x", "<system>agent</system>", "<cot>plan</cot>", "<diff>--- a\n+++ b</diff>"] * 8,
        target_size=128, dim=args.dim, ast_dim=8,
    )
    arch = HybridConfig(dim=args.dim, n_layers=args.layers)
    qat_cfg = QATConfig(device="cpu", lora_rank=4, vram_bytes=4 * 1024 ** 3)
    eng_cfg = EngineConfig(device="cpu", seq_len=args.seq_len, batch_size=1)
    global _ENGINE
    _ENGINE = Engine(vocab, arch, qat_cfg, eng_cfg)

    print("[train] ccwatch monitor: run `npx ccwatch` in another terminal to watch cost/quota live")

    losses = []
    for step in range(args.steps):
        rows = _collect_ccsniff_rows()
        if not rows:
            print(f"[train] step {step}: no ccsniff data yet")
            continue
        loss = _train_on_rows(rows, args.batch)
        losses.append(loss)
        print(f"[train] step {step}: rows={len(rows)} loss={loss:.4f}")

    if losses:
        print(f"[train] final avg loss={sum(losses)/len(losses):.4f}")
    _ENGINE.close()
    print("[train] done")


if __name__ == "__main__":
    raise SystemExit(main())
