#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
VENV="$(pwd)/.venv"

echo "== [1/7] Seed Claude Code session =="
"$VENV/bin/python" scripts/seed_claude_session.py

echo ""
echo "== [2/7] Verify npx ccsniff sees the session =="
npx --yes ccsniff@latest --list-sessions 2>&1 | tail -3

echo ""
echo "== [3/7] Run tianji demo =="
"$VENV/bin/python" -m tianji.cli demo 2>&1 | tail -12

echo ""
echo "== [4/7] Ingest via npx ccsniff -> tianji =="
"$VENV/bin/python" -m tianji.cli ingest-ccsniff 2>&1 | tail -5

echo ""
echo "== [5/7] Generate tokens =="
"$VENV/bin/python" -m tianji.cli infer --prompt "def fib(n): return n" --n 8 2>&1 | tail -3

echo ""
echo "== [6/7] Checkpoint save/load =="
"$VENV/bin/python" -m tianji.cli checkpoint save /tmp/tianji-ckpt.pt 2>&1 | tail -2
"$VENV/bin/python" -m tianji.cli checkpoint load /tmp/tianji-ckpt.pt 2>&1 | tail -2

echo ""
echo "== [7/7] Run tests =="
"$VENV/bin/python" -m pytest python/tests/ -q 2>&1 | tail -5

echo ""
echo "== pipeline complete =="
