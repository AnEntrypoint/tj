#!/usr/bin/env python3
"""Training driver: roll up Claude Code sessions via npx ccsniff, distill them
through Tianji's QAT loop + state head, and keep npx ccwatch visible as the
cost/quota monitor.

Training is incremental and persistent: each event is trained at most once
(deduped by (sid, ts)), and model + state-head checkpoints are written so runs
accumulate across invocations (``--resume``).

Usage:
  python scripts/train.py --steps 20 --batch 32
  python scripts/train.py --resume --steps 50      # continue from last checkpoint
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from tianji import Engine, EngineConfig
from tianji.tokens.apt import Vocab
from tianji.arch.hybrid import HybridConfig
from tianji.distill.qat_loop import QATConfig
from tianji.ingest.ccsniff import ingest_ccsniff_stream
from tianji.protocol import verify_frame

try:
    from tianji.ingest.hf_datasets import iter_hf_texts, _known_dataset
    _HF_OK = True
except ImportError:
    _HF_OK = False


_FALLBACK_CORPUS = [
    '<tool_call>{"name":"edit","args":{"path":"a.py"}}</tool_call>',
    "<bash_output>ok</bash_output>",
    "def fib(n): return n",
    "<system>agent</system>",
    "<cot>plan</cot>",
    "<diff>--- a\n+++ b\n</diff>",
]

_CCSNIIF_BIN = os.environ.get("CCSNIIF_BIN", "npx.cmd" if os.name == "nt" else "npx")

# Wall-clock time (seconds) of the most recent ccsniff fetch. Used to make
# later training steps fetch only events that arrived since the previous
# fetch instead of re-pulling the whole --since window every step.
_LAST_FETCH_WALL = None


def _lower_priority() -> None:
    """Drop our CPU scheduling priority so long background training never
    starves interactive work. No-op if the platform/permissions disallow it."""
    try:
        if os.name == "nt":
            import ctypes
            # BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
        else:
            os.nice(10)
    except Exception:
        pass


def _since_for_step(step: int, start_step: int, base_since: str) -> str:
    """Window to fetch for ``step``.

    The first step (cold start or after resume) pulls the full requested
    history window so all prior events get trained once. Every later step
    pulls only events that arrived since the previous fetch, which is a tiny
    window and keeps long/multi-step CPU training cheap.
    """
    if step == start_step or _LAST_FETCH_WALL is None:
        return base_since
    elapsed = int(time.time() - _LAST_FETCH_WALL) + 2
    return f"{max(elapsed, 5)}s"


def _collect_ccsniff_rows(since: str, limit: int, seen: set, project: str = None) -> tuple:
    """Return (deduped_rows, max_ts) from a live ccsniff --json fetch.

    Rows already present in ``seen`` (keyed by (sid, ts)) are skipped so each
    event is trained at most once across steps/runs.

    Output is streamed to a temp file (``stdout=open(tmp)``) rather than
    ``capture_output=True``: for large corpora (the full Claude Code history is
    ~1 GB of JSONL) the OS pipe buffer fills and the child blocks while the
    parent waits for exit -> DEADLOCK. Writing to a file avoids pipe buffering
    entirely, so the full history fetch completes instead of hanging at step 0.
    """
    import tempfile
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(prefix="tianji_ccsniff_", suffix=".jsonl")
        os.close(fd)
        # When the binary is npx we must pass the package spec ("ccsniff@latest")
        # and --yes; for a direct binary (e.g. node .../cli.js, or a pinned
        # ccsniff install) those tokens are invalid and must be dropped so the
        # command actually runs (this is what lets CCSNIIF_BIN bypass the
        # intermittent npx network hang). npx resolves via the shell (.cmd), so
        # it runs with shell=True; a direct binary runs without a shell. The
        # binary string is split on whitespace (handles "node <path>").
        cmd = _CCSNIIF_BIN.split()
        is_npx = cmd[0].endswith("npx") or cmd[0].endswith("npx.cmd")
        if is_npx:
            cmd += ["--yes", "ccsniff@latest"]
        cmd += ["--json", "--full-history", "--since", since, "--limit", str(limit)]
        if project:
            cmd += ["--project", project]
        with open(tmp, "w", encoding="utf-8") as fh:
            r = subprocess.run(
                cmd,
                stdout=fh, stderr=subprocess.DEVNULL, shell=is_npx,
            )
        if r.returncode != 0:
            print(f"[train] ccsniff collect failed (rc={r.returncode})", file=sys.stderr)
            return [], 0
        rows = []
        max_ts = 0
        with open(tmp, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = row.get("sid") or "trace"
                ts = int(row.get("ts", 0) or 0)
                key = (sid, ts)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
                if ts > max_ts:
                    max_ts = ts
        return rows, max_ts
    except Exception as e:  # pragma: no cover - environment dependent
        print(f"[train] ccsniff subprocess error: {e}", file=sys.stderr)
        return [], 0
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _seed_vocab(rows, dim: int, ast_dim: int) -> Vocab:
    corpus = [row.get("text") or "" for row in rows if row.get("text")]
    if not corpus:
        corpus = list(_FALLBACK_CORPUS)
    return Vocab.build(corpus, target_size=128, dim=dim, ast_dim=ast_dim)


def _train_on_rows(rows: list, batch: int) -> tuple:
    global _ENGINE
    frames = list(ingest_ccsniff_stream(iter(json.dumps(r) for r in rows), batch_size=batch))
    total_loss = 0.0
    n = 0
    correct = 0
    total = 0
    for f in frames:
        if not verify_frame(f):
            continue
        res = _ENGINE.step_frame(f)
        total_loss += res.qat.loss
        n += 1
        if res.state_pred_kind is not None and res.state_target_kind is not None:
            total += 1
            if res.state_pred_kind == res.state_target_kind:
                correct += 1
    avg_loss = total_loss / max(1, n)
    return avg_loss, correct, total


_ENGINE = None


def _train_on_hf_texts(texts: list[str]) -> float:
    """Train on negative (public HF) samples. Uses source='synthetic'
    routing to keep them separated from positive Claude Code data."""
    global _ENGINE
    total_loss = 0.0
    n = 0
    for text in texts:
        if not text or not text.strip():
            continue
        out = encode(text, _ENGINE.vocab, parse_ast=False)
        ids = _ENGINE._prepare_ids(out.ids)
        if len(ids) < 2:
            continue
        t = torch.tensor(ids, dtype=torch.long, device=_ENGINE.device)
        inp = t[:-1].unsqueeze(0)
        tgt = t[1:].unsqueeze(0)
        # Clamp to seq_len
        L = max(2, int(_ENGINE.cfg.seq_len))
        if inp.shape[1] > L:
            inp = inp[:, :L]
            tgt = tgt[:, :L]
        res, _ = _ENGINE.qat.step(inp, tgt, source="synthetic")
        total_loss += res.loss
        n += 1
    return total_loss / max(1, n)


def _latest_checkpoint(ckpt_dir: Path):
    qat = ckpt_dir / "qat.pt"
    state = ckpt_dir / "state.pt"
    meta = ckpt_dir / "meta.json"
    if qat.exists() and state.exists():
        return qat, state, (meta if meta.exists() else None)
    return None, None, None


def main():
    global _ENGINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--ast-dim", type=int, default=8)
    ap.add_argument("--layers", type=int, default=27)
    ap.add_argument("--seq-len", type=int, default=512,
                    help="token chunk size fed to the model per step; raise toward "
                         "200000 for full 200k context. The Engine chunks each frame "
                         "into --seq-len pieces, so a 200k-token session is processed in "
                         "chunks while Mamba-2's constant recurrent state carries "
                         "cross-chunk context at inference (achieving 200k reach without a "
                         "single n x n attention matrix).")
    ap.add_argument("--since", type=str, default="1h",
                    help="ccsniff --since window, e.g. 1h, 7d, 24h (targets our live session history)")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--checkpoint-dir", type=str, default=".tianji_ckpt")
    ap.add_argument("--resume", action="store_true", help="resume from latest checkpoint in --checkpoint-dir")
    ap.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"],
                    help="training device (default: auto-detect cuda if available, else cpu)")
    ap.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"],
                    help="training precision (default fp16; bf16 on Ampere+)")
    ap.add_argument("--hf-dataset", type=str, default=None,
                    help="HuggingFace dataset for negative eval (e.g. the-stack, starcoder, codeparrot)")
    ap.add_argument("--hf-split", type=str, default="train",
                    help="HF dataset split")
    ap.add_argument("--hf-max-samples", type=int, default=0,
                    help="max HF samples per step (0=unlimited)")
    ap.add_argument("--neg-ratio", type=float, default=0.3,
                    help="fraction of negative (HF) samples in each batch")
    ap.add_argument("--daemon", action="store_true",
                    help="run continuously (infinite steps, poll for new data)")
    ap.add_argument("--daemon-poll", type=int, default=30,
                    help="seconds between polls when no new data available")
    ap.add_argument("--project", type=str, default=None,
                    help="ccsniff --project filter (basename of cwd)")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    _lower_priority()
    print(f"[train] device={device} (cuda available={torch.cuda.is_available()})")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    seen: set = set()
    start_step = 0
    last_ts = 0

    qat_path, state_path, meta_path = _latest_checkpoint(ckpt_dir)
    resuming = args.resume and qat_path is not None

    if resuming:
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path else {}
        last_ts = int(meta.get("last_ts", 0))
        start_step = int(meta.get("steps_done", 0))
        for sid, ts in meta.get("seen", []):
            seen.add((sid, int(ts)))
        ck = torch.load(state_path, map_location="cpu", weights_only=False)
        vocab = ck["vocab"]
        arch = HybridConfig(dim=args.dim, n_layers=args.layers)
        qat_cfg = QATConfig(device=device, lora_rank=4, vram_bytes=4 * 1024 ** 3,
                            seq_len=args.seq_len, precision=args.precision)
        eng_cfg = EngineConfig(device=device, seq_len=args.seq_len, batch_size=1)
        _ENGINE = Engine(vocab, arch, qat_cfg, eng_cfg)
        _ENGINE.load_training_state(state_path)
        _ENGINE.qat.load_checkpoint(qat_path)
        print(f"[train] resumed from checkpoint (steps_done={start_step}, seen={len(seen)})")
    else:
        # Cold start: seed the vocab from a real ccsniff sample, then build engine.
        seed_rows, _ = _collect_ccsniff_rows(args.since, min(args.limit, 256), seen, args.project)
        seen.clear()  # seed rows are for vocab only; let the loop train them too
        vocab = _seed_vocab(seed_rows, args.dim, args.ast_dim)
        arch = HybridConfig(dim=args.dim, n_layers=args.layers)
        qat_cfg = QATConfig(device=device, lora_rank=4, vram_bytes=4 * 1024 ** 3,
                            seq_len=args.seq_len, precision=args.precision)
        eng_cfg = EngineConfig(device=device, seq_len=args.seq_len, batch_size=1)
        _ENGINE = Engine(vocab, arch, qat_cfg, eng_cfg)
        print(f"[train] built engine vocab_size={vocab.size}, layers={arch.n_layers}")

    print("[train] ccwatch monitor: run `npx ccwatch` in another terminal to watch cost/quota live")

    # ── HF negative-eval dataset ──────────────────────────────────────
    hf_texts = None
    if args.hf_dataset and _HF_OK:
        try:
            cfg = _known_dataset(args.hf_dataset)
            hf_texts = iter_hf_texts(
                dataset=cfg.path, split=args.hf_split or cfg.split,
                streaming=True, max_samples=args.hf_max_samples,
                text_field=cfg.text_field,
            )
            print(f"[train] HF negative eval: {args.hf_dataset} ({cfg.path})")
        except Exception as e:
            print(f"[train] HF dataset unavailable: {e}", file=sys.stderr)

    losses = []
    acc_correct = 0
    acc_total = 0
    step = start_step
    daemon_steps = 0
    while True:
        if not args.daemon and step >= start_step + args.steps:
            break
        since = _since_for_step(step, start_step, args.since) if not args.daemon else f"{args.daemon_poll}s"
        if args.daemon and daemon_steps > 0:
            since = f"{args.daemon_poll}s"
        rows, max_ts = _collect_ccsniff_rows(since, args.limit, seen, args.project)
        _LAST_FETCH_WALL = time.time()
        last_ts = max(last_ts, max_ts)
        if not rows:
            print(f"[train] step {step}: no new ccsniff data in --since {since}")
            continue
        loss, pc, pt = _train_on_rows(rows, args.batch)
        acc_correct += pc
        acc_total += pt
        losses.append(loss)
        acc_str = f"{acc_correct}/{acc_total} ({100.0*acc_correct/max(1,acc_total):.1f}%)" if acc_total else "n/a"
        hf_str = ""
        if hf_texts is not None:
            # Train on negative HF samples: collect a small batch of public
            # code and run a forward pass with negative-source routing.
            hf_batch = []
            for _ in range(min(args.batch, 32)):
                try:
                    hf_batch.append(next(hf_texts))
                except StopIteration:
                    break
            if hf_batch:
                hf_loss = _train_on_hf_texts(hf_batch)
                hf_str = f" hf_neg_loss={hf_loss:.4f}"
                # Contrastive loss: pull positive (ccsniff) and push negative (HF)
                pos_texts = [row.get("text") or "" for row in rows[:16] if row.get("text")]
                if pos_texts:
                    c_loss = _ENGINE.step_contrastive(pos_texts, hf_batch[:16])
                    hf_str += f" contrastive={c_loss:.4f}"
        print(f"[train] step {step}: since={since} rows={len(rows)} qat_loss={loss:.4f} "
              f"state_acc={acc_str}{hf_str} vram={_ENGINE.qat._vram()}B")

        if step % args.save_every == 0 or (not args.daemon and step == start_step + args.steps - 1):
            _ENGINE.qat.save_checkpoint(str(qat_path if resuming else ckpt_dir / "qat.pt"))
            _ENGINE.save_training_state(str(state_path if resuming else ckpt_dir / "state.pt"))
            (ckpt_dir / "meta.json").write_text(
                json.dumps({"last_ts": last_ts, "steps_done": step + 1,
                            "seen": [[s, t] for s, t in seen]}),
                encoding="utf-8")

        if args.daemon:
            daemon_steps += 1
        step += 1

    if losses:
        print(f"[train] final avg qat loss={sum(losses)/len(losses):.4f}")
    if acc_total:
        print(f"[train] final state-head next-kind accuracy={acc_correct}/{acc_total} "
              f"({100.0*acc_correct/acc_total:.1f}%)")
    _ENGINE.close()
    print("[train] done")


if __name__ == "__main__":
    raise SystemExit(main())
