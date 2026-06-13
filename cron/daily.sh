#!/usr/bin/env bash
# Daily 05:45 — sync free model list + probe liveness
# Crontab: 45 5 * * * /home/alain/free-llm-api/cron/daily.sh >> /tmp/llm-pool-daily.log 2>&1

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== $(date -u '+%Y-%m-%dT%H:%M:%SZ') daily sync ==="

# Load API keys
# shellcheck source=/dev/null
source ~/.profile

PYTHON=./venv/bin/python3

echo "--- cheahjs sync ---"
$PYTHON discovery/cheahjs_sync.py

echo "--- liveness probe ---"
$PYTHON health/liveness.py

echo "--- pool summary ---"
$PYTHON - <<'EOF'
from router.pool import summary, candidates
import sys
pool = candidates()
alive = [m for m in pool if m.get('alive') is True]
print(summary())
if not alive:
    print("WARNING: no alive models in pool", file=sys.stderr)
    sys.exit(1)
EOF

echo "=== done ==="
