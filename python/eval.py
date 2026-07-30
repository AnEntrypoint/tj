#!/usr/bin/env python3
"""Tianji model evaluation — measures how good the model is at any stage.

Run during training to track progress, or standalone to evaluate a checkpoint.

Usage:
    python eval.py                              # eval latest checkpoint
    python eval.py --ckpt .tianji_ckpt/qat.pt   # eval specific checkpoint
    python eval.py --watch                      # watch eval metrics during training
    python eval.py --compare a.pt b.pt          # compare two checkpoints
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from tianji.tokens.apt import Vocab, encode, decode
from tianji.arch.hybrid import HybridConfig
from tianji.distill.qat_loop import QATLoop, QATConfig
from tianji.infer.generator import Generator, GenerateConfig, SamplingConfig
from tianji.engine import Engine, EngineConfig
from tianji.protocol import (
    Frame, Trajectory, ToolCall, ToolResult, frame_hash, make_frame, verify_frame,
)


def _build_eval_data() -> tuple[list[str], list[str]]:
    """Build a small held-out eval set. Returns (positives, negatives)."""
    positives = [
        "<system>You are Claude Code, an AI coding assistant.</system>",
        "<cot>I need to read the file to understand the current implementation.</cot>",
        '<tool_call>{"name":"Read","args":{"path":"src/main.py"}}</tool_call>',
        "<bash_output>def main():\n    print('hello')\n</bash_output>",
        "<diff>--- a/file.py\n+++ b/file.py\n@@ -1,3 +1,4 @@\n import os\n+import sys\n</diff>",
        "I'll fix that bug by updating the error handling.",
        "<system>agent</system>\n<cot>Let me check the test results first.</cot>",
        '<tool_call>{"name":"Bash","args":{"command":"pytest tests/ -q"}}</tool_call>',
        "<bash_output>64 passed in 2.34s</bash_output>",
        "<cot>The tests pass. Now I'll implement the feature.</cot>",
    ]
    negatives = [
        "def add(a, b): return a + b",
        "class Foo: pass",
        "import os\nimport sys\n\nprint('hello world')",
        "for i in range(10):\n    print(i)",
        "x = [1, 2, 3]\ny = [4, 5, 6]",
        "The quick brown fox jumps over the lazy dog",
        "Lorem ipsum dolor sit amet",
        "def factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n-1)",
        "try:\n    x = 1/0\nexcept: pass",
        "with open('f.txt') as f: data = f.read()",
    ]
    return positives, negatives


def eval_model(ckpt_path: str, dim: int = 768, device: str = "cuda") -> dict:
    """Evaluate a trained model checkpoint. Returns dict of metrics."""
    results = {}
    vocab = Vocab.build(["test"] * 10, target_size=128, dim=dim, ast_dim=8)
    arch = HybridConfig(dim=dim, n_layers=27)
    cfg = QATConfig(device=device, lora_rank=4, vram_bytes=4 * 1024 ** 3,
                    precision="fp16" if device == "cuda" else "fp32")
    qat = QATLoop(cfg, arch, vocab_size=vocab.size)

    if os.path.exists(ckpt_path):
        n = qat.load_checkpoint(ckpt_path)
        results["checkpoint"] = ckpt_path
        results["adapters_loaded"] = n
    else:
        results["checkpoint"] = "none (untrained)"
        results["adapters_loaded"] = 0

    pos_texts, neg_texts = _build_eval_data()

    # ── 1. Perplexity on eval set ────────────────────────────────────
    total_loss = 0.0
    total_tokens = 0
    for text in pos_texts + neg_texts:
        out = encode(text, vocab, parse_ast=False)
        ids = out.ids
        if len(ids) < 2:
            continue
        t = torch.tensor(ids, dtype=torch.long, device=qat.device)
        inp = t[:-1].unsqueeze(0)
        tgt = t[1:].unsqueeze(0)
        L = 1024
        if inp.shape[1] > L:
            inp = inp[:, :L]
            tgt = tgt[:, :L]
        with torch.no_grad():
            logits, _, _ = qat.model(inp)
            loss = F.cross_entropy(
                logits.view(-1, vocab.size), tgt.view(-1), reduction="sum")
            total_loss += loss.item()
            total_tokens += tgt.numel()
    perplexity = torch.exp(torch.tensor(total_loss / max(1, total_tokens))).item()
    results["perplexity"] = round(perplexity, 2)
    results["eval_tokens"] = total_tokens

    # ── 2. Positive vs negative loss separation ──────────────────────
    pos_loss = 0.0
    neg_loss = 0.0
    pos_tokens = 0
    neg_tokens = 0
    for text in pos_texts:
        out = encode(text, vocab, parse_ast=False)
        ids = out.ids
        if len(ids) < 2:
            continue
        t = torch.tensor(ids, dtype=torch.long, device=qat.device)
        inp = t[:-1].unsqueeze(0)[:, :1024]
        tgt = t[1:].unsqueeze(0)[:, :1024]
        with torch.no_grad():
            logits, _, _ = qat.model(inp)
            pos_loss += F.cross_entropy(
                logits.view(-1, vocab.size), tgt.view(-1), reduction="sum").item()
            pos_tokens += tgt.numel()
    for text in neg_texts:
        out = encode(text, vocab, parse_ast=False)
        ids = out.ids
        if len(ids) < 2:
            continue
        t = torch.tensor(ids, dtype=torch.long, device=qat.device)
        inp = t[:-1].unsqueeze(0)[:, :1024]
        tgt = t[1:].unsqueeze(0)[:, :1024]
        with torch.no_grad():
            logits, _, _ = qat.model(inp)
            neg_loss += F.cross_entropy(
                logits.view(-1, vocab.size), tgt.view(-1), reduction="sum").item()
            neg_tokens += tgt.numel()
    pos_ppl = torch.exp(torch.tensor(pos_loss / max(1, pos_tokens))).item()
    neg_ppl = torch.exp(torch.tensor(neg_loss / max(1, neg_tokens))).item()
    results["positive_perplexity"] = round(pos_ppl, 2)
    results["negative_perplexity"] = round(neg_ppl, 2)
    # Higher ratio = better separation (model finds positives easier)
    results["pos_neg_ratio"] = round(neg_ppl / max(1, pos_ppl), 2)

    # ── 3. Generation quality ────────────────────────────────────────
    gen = Generator(qat, GenerateConfig(max_tokens=32, sampling=SamplingConfig(temperature=0.7)))
    prompts = [
        "<system>",
        "<cot>",
        "<tool_call>",
        "def ",
        "import ",
    ]
    generations = []
    for prompt in prompts:
        out = encode(prompt, vocab, parse_ast=False)
        if not out.ids:
            continue
        steps = list(gen.generate(out.ids))
        text = decode([s.token for s in steps], vocab)
        # Count special tokens in output (proxy for agent-like behavior)
        special_count = sum(
            1 for tag in ["<tool_call>", "</tool_call>", "<cot>", "</cot>",
                          "<system>", "</system>", "<diff>", "</diff>",
                          "<bash_output>", "</bash_output>"]
            if tag in text
        )
        generations.append({"prompt": prompt, "output": text[:80], "special_tokens": special_count})
    results["generations"] = generations
    avg_special = sum(g["special_tokens"] for g in generations) / max(1, len(generations))
    results["avg_special_tokens"] = round(avg_special, 1)

    # ── 4. Embedding separation ──────────────────────────────────────
    pos_embeds = []
    neg_embeds = []
    for text in pos_texts[:5]:
        out = encode(text, vocab, parse_ast=False)
        ids = out.ids
        if len(ids) < 2:
            continue
        t = torch.tensor(ids, dtype=torch.long, device=qat.device).unsqueeze(0)
        with torch.no_grad():
            h = qat.hidden(t)
            pos_embeds.append(h[:, -1].cpu())  # last-token pooling
    for text in neg_texts[:5]:
        out = encode(text, vocab, parse_ast=False)
        ids = out.ids
        if len(ids) < 2:
            continue
        t = torch.tensor(ids, dtype=torch.long, device=qat.device).unsqueeze(0)
        with torch.no_grad():
            h = qat.hidden(t)
            neg_embeds.append(h[:, -1].cpu())
    if pos_embeds and neg_embeds:
        pos_t = torch.cat(pos_embeds, dim=0)
        neg_t = torch.cat(neg_embeds, dim=0)
        # Cosine similarity within and across groups
        pos_sim = F.cosine_similarity(pos_t.unsqueeze(1), pos_t.unsqueeze(0), dim=-1)
        neg_sim = F.cosine_similarity(neg_t.unsqueeze(1), neg_t.unsqueeze(0), dim=-1)
        cross_sim = F.cosine_similarity(pos_t.unsqueeze(1), neg_t.unsqueeze(0), dim=-1)
        # Exclude self-similarity
        mask = ~torch.eye(pos_sim.shape[0], dtype=torch.bool)
        results["pos_intra_similarity"] = round(pos_sim[mask].mean().item(), 4)
        results["neg_intra_similarity"] = round(neg_sim[mask].mean().item(), 4)
        results["cross_similarity"] = round(cross_sim.mean().item(), 4)
        # Separation = intra - cross (higher = better)
        results["embedding_separation"] = round(
            (pos_sim[mask].mean().item() + neg_sim[mask].mean().item()) / 2
            - cross_sim.mean().item(), 4
        )

    # ── 5. VRAM ──────────────────────────────────────────────────────
    results["vram_mb"] = round(qat._vram() / 1024**2, 1)
    qat.close()
    return results


def print_eval(results: dict) -> None:
    """Pretty-print eval results."""
    print(f"\n{'='*50}")
    print(f"  Tianji Model Evaluation")
    print(f"{'='*50}")
    print(f"  Checkpoint:  {results.get('checkpoint', 'N/A')}")
    print(f"  Adapters:    {results.get('adapters_loaded', 0)}")
    print(f"  VRAM:        {results.get('vram_mb', 0)} MB")
    print(f"\n  ── Language Quality ──")
    print(f"  Perplexity:          {results.get('perplexity', 'N/A')}")
    print(f"  Positive perplexity: {results.get('positive_perplexity', 'N/A')}")
    print(f"  Negative perplexity: {results.get('negative_perplexity', 'N/A')}")
    print(f"  Pos/Neg ratio:       {results.get('pos_neg_ratio', 'N/A')} (>1 = model prefers positives)")
    print(f"\n  ── Generation Quality ──")
    print(f"  Avg special tokens:  {results.get('avg_special_tokens', 'N/A')} (higher = more agent-like)")
    for g in results.get("generations", [])[:3]:
        special = "✓" if g["special_tokens"] > 0 else " "
        print(f"  [{special}] '{g['prompt']}' → '{g['output'][:60]}'")
    print(f"\n  ── Embedding Separation ──")
    print(f"  Pos intra-sim:  {results.get('pos_intra_similarity', 'N/A')}")
    print(f"  Neg intra-sim:  {results.get('neg_intra_similarity', 'N/A')}")
    print(f"  Cross-sim:      {results.get('cross_similarity', 'N/A')}")
    print(f"  Separation:     {results.get('embedding_separation', 'N/A')} (>0 = separated)")
    print(f"{'='*50}\n")


def main():
    ap = argparse.ArgumentParser(description="Tianji model evaluation")
    ap.add_argument("--ckpt", default=".tianji_ckpt/qat.pt", help="checkpoint path")
    ap.add_argument("--dim", type=int, default=768, help="model dimension")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--compare", nargs=2, default=None, help="compare two checkpoints")
    ap.add_argument("--json", action="store_true", help="output JSON only")
    args = ap.parse_args()

    if args.compare:
        print("Comparing checkpoints...")
        for ckpt in args.compare:
            print(f"\n  {ckpt}:")
            r = eval_model(ckpt, dim=args.dim, device=args.device)
            if args.json:
                print(json.dumps(r, indent=2))
            else:
                print_eval(r)
        return

    results = eval_model(args.ckpt, dim=args.dim, device=args.device)
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_eval(results)


if __name__ == "__main__":
    main()