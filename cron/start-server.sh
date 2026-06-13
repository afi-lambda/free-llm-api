#!/usr/bin/env bash
# Start the OpenAI-compatible proxy server.
# Systemd service: ~/.config/systemd/user/free-llm-pool.service
# Manual:          ./cron/start-server.sh
# Env override:    FREE_LLM_PORT=8001 ./cron/start-server.sh

set -euo pipefail
cd "$(dirname "$0")/.."

source ~/.profile

PORT="${FREE_LLM_PORT:-8000}"

exec ./venv/bin/python3 -m uvicorn router.server:app \
    --host 127.0.0.1 \
    --port "$PORT" \
    --log-level warning
