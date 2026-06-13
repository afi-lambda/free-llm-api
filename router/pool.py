"""
Ordered candidate pool: reads registry.json and returns usable models by priority.
"""

import json
import os
from datetime import date
from pathlib import Path

REGISTRY = Path(__file__).parent.parent / "registry.json"
MAX_LAST_SEEN_DAYS = 2


def _has_key(model: dict) -> bool:
    return bool(os.environ.get(model.get("env_key", "")))


def _last_seen_ok(model: dict) -> bool:
    ls = model.get("last_seen")
    if not ls:
        return True
    try:
        return (date.today() - date.fromisoformat(ls)).days <= MAX_LAST_SEEN_DAYS
    except ValueError:
        return True


def candidates(require_alive: bool = True) -> list[dict]:
    """Return models sorted by (tier, -swebench_score, latency_ms), filtered to usable ones."""
    if not REGISTRY.exists():
        return []
    registry = json.loads(REGISTRY.read_text())
    models = registry.get("models", [])

    def usable(m: dict) -> bool:
        if not _has_key(m):
            return False
        if not _last_seen_ok(m):
            return False
        if require_alive and m.get("alive") is False:
            return False
        return True

    usable_models = [m for m in models if usable(m)]
    usable_models.sort(key=lambda m: (
        m.get("tier", 3),
        -m.get("swebench_score", 0.0),
        m.get("latency_ms") or 9999,
    ))
    return usable_models


def best() -> dict | None:
    pool = candidates()
    return pool[0] if pool else None


def summary() -> str:
    pool = candidates()
    if not pool:
        return "Pool empty"
    top = pool[0]
    return (
        f"{len(pool)} candidates — best: {top['id']} "
        f"(T{top.get('tier','?')}, score={top.get('swebench_score', 0):.0%})"
    )
