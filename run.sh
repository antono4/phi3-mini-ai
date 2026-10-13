#!/usr/bin/env bash
# Start the Phi-3 Mini web server (chat UI + REST API) on port 12000.
set -u
cd "$(dirname "$0")"
MODEL="${PHI3_MODEL_PATH:-models/Phi-3-mini-4k-instruct-q4.gguf}"
PY="python3"
[ -x .venv/bin/python ] && PY=".venv/bin/python"
if [ ! -f "$MODEL" ]; then
  echo "Model missing at $MODEL. Downloading Q4..."
  "$PY" scripts/download_model.py
fi
export PHI3_PORT="${PHI3_PORT:-12000}"
export PATH="$HOME/.local/bin:$PATH"
echo "Starting Phi-3 Mini server on http://0.0.0.0:${PHI3_PORT}"
exec "$PY" -m uvicorn phi3.server:app --host 0.0.0.0 --port "$PHI3_PORT"
