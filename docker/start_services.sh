#!/usr/bin/env bash
set -euo pipefail

BITNET_DIR="${BITNET_DIR:-/opt/bitnet}"
BITNET_SERVER_SCRIPT="${BITNET_SERVER_SCRIPT:-run_inference_server.py}"
BITNET_MODEL="${BITNET_MODEL:-models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf}"
BITNET_THREADS="${BITNET_THREADS:-6}"
BITNET_PORT="${BITNET_PORT:-8080}"
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"

if [ ! -d "$BITNET_DIR" ]; then
  echo "ERROR: BITNET_DIR does not exist: $BITNET_DIR"
  echo "Mount your BitNet folder, e.g. -v /host/path/to/BitNet:$BITNET_DIR"
  exit 1
fi

if [ ! -f "$BITNET_DIR/$BITNET_SERVER_SCRIPT" ]; then
  echo "ERROR: BitNet server script not found: $BITNET_DIR/$BITNET_SERVER_SCRIPT"
  exit 1
fi

if [ ! -f "$BITNET_DIR/$BITNET_MODEL" ]; then
  echo "ERROR: BitNet model file not found: $BITNET_DIR/$BITNET_MODEL"
  exit 1
fi

cd "$BITNET_DIR"
echo "Starting BitNet server from: $BITNET_DIR"
python3 "$BITNET_SERVER_SCRIPT" -m "$BITNET_MODEL" -t "$BITNET_THREADS" --port "$BITNET_PORT" &
BITNET_PID=$!

cleanup() {
  if kill -0 "$BITNET_PID" 2>/dev/null; then
    kill "$BITNET_PID" || true
    wait "$BITNET_PID" || true
  fi
}
trap cleanup EXIT INT TERM

echo "Waiting for BitNet on port $BITNET_PORT..."
for _ in $(seq 1 120); do
  if ! kill -0 "$BITNET_PID" 2>/dev/null; then
    echo "ERROR: BitNet server exited during startup."
    exit 1
  fi
  if curl --fail --silent "http://127.0.0.1:$BITNET_PORT/health" >/dev/null 2>&1; then
    echo "BitNet is up."
    break
  fi
  sleep 1
done

if ! curl --fail --silent "http://127.0.0.1:$BITNET_PORT/health" >/dev/null 2>&1; then
  echo "ERROR: BitNet did not start on port $BITNET_PORT in time."
  exit 1
fi

cd /app
echo "Starting API on $API_HOST:$API_PORT"
python main.py &
API_PID=$!
while kill -0 "$BITNET_PID" 2>/dev/null && kill -0 "$API_PID" 2>/dev/null; do
  sleep 1
done
if kill -0 "$API_PID" 2>/dev/null; then
  kill "$API_PID" || true
  wait "$API_PID" || true
  echo "ERROR: BitNet server exited."
  exit 1
fi
wait "$API_PID"
