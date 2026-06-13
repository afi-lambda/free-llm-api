#!/usr/bin/env bash
# Weekly Monday 05:45 — refresh SWE-bench scores + run smoke tests
# Runs AFTER daily.sh (which handles sync + liveness), so models are already probed.
# Crontab: 45 5 * * 1 /home/alain/free-llm-api/cron/weekly.sh >> /tmp/llm-pool-weekly.log 2>&1

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== $(date -u '+%Y-%m-%dT%H:%M:%SZ') weekly quality check ==="

# Load API keys
# shellcheck source=/dev/null
source ~/.profile

PYTHON=./venv/bin/python3

echo "--- SWE-bench score refresh ---"
$PYTHON discovery/swebench.py

echo "--- smoke test (Tier 1+2 + 1 rotating Tier 3) ---"
$PYTHON health/smoke.py

echo "=== done ==="
