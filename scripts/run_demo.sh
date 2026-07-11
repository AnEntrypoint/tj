#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"

echo "== [1/7] Verify npx ccwatch is installed (cost/quota monitor) =="
npx --yes @terzigolu/ccwatch version 2>&1 | tail -2

echo ""
echo "== [2/7] Verify npx ccsniff sees sessions =="
npx --yes ccsniff@latest --list-sessions 2>&1 | tail -3

echo ""
echo "== [3/7] Run tianji demo =="
"$PY" -m tianji.cli demo 2>&1 | tail -12

echo ""
echo "== [4/7] Ingest via npx ccsniff -> tianji =="
"$PY" -m tianji.cli ingest-ccsniff 2>&1 | tail -5

echo ""
echo "== [5/7] Generate tokens =="
"$PY" -m tianji.cli infer --prompt "def fib(n): return n" --n 8 2>&1 | tail -3

echo ""
echo "== [6/7] Checkpoint save/load =="
"$PY" -m tianji.cli checkpoint save /tmp/tianji-ckpt.pt 2>&1 | tail -2
"$PY" -m tianji.cli checkpoint load /tmp/tianji-ckpt.pt 2>&1 | tail -2

echo ""
echo "== [7/7] Run tests =="
"$PY" -m pytest python/tests/ -q 2>&1 | tail -5

echo ""
echo "== pipeline complete =="
echo "== to train continuously: python scripts/train.py --steps 20 (run 'npx ccwatch' in another terminal to monitor cost) =="
