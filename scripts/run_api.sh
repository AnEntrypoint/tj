#!/usr/bin/env bash
# Tianji API server launcher
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
HOST="${TIANJI_HOST:-0.0.0.0}"
PORT="${TIANJI_PORT:-8080}"
DEVICE="${TIANJI_DEVICE:-cpu}"
CKPT="${TIANJI_CKPT:-}"
API_KEY="${TIANJI_API_KEY:-}"

echo "== Tianji API server =="
echo "  host:   $HOST"
echo "  port:   $PORT"
echo "  device: $DEVICE"
echo "  ckpt:   ${CKPT:-(none)}"

ARGS="--host $HOST --port $PORT --device $DEVICE"
if [ -n "$CKPT" ]; then
  ARGS="$ARGS --ckpt $CKPT"
fi
if [ -n "$API_KEY" ]; then
  export TIANJI_API_KEY="$API_KEY"
  echo "  auth:   api-key enabled"
fi

exec "$PY" -m tianji.server $ARGS
