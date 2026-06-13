"""
Per-model request budget tracker.
Persists to disk so restarts don't lose window state.
"""

import json
import time
from datetime import date, datetime, timezone
from pathlib import Path
from threading import Lock

STATE_FILE = Path(__file__).parent.parent / ".rate_state.json"
_lock = Lock()


def _load() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"date": str(date.today()), "daily": {}, "limited_until": {}}


def _save(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state))


def _fresh_day(state: dict) -> dict:
    today = str(date.today())
    if state.get("date") != today:
        return {"date": today, "daily": {}, "limited_until": state.get("limited_until", {})}
    return state


class RateTracker:
    """Thread-safe, file-backed budget tracker."""

    def __init__(self) -> None:
        with _lock:
            self._state = _fresh_day(_load())

    def _sync(self) -> dict:
        self._state = _fresh_day(self._state)
        return self._state

    def is_exhausted(self, model: dict) -> bool:
        with _lock:
            state = self._sync()
            model_id = model["id"]

            # Check explicit rate-limit hold
            hold_until = state["limited_until"].get(model_id, 0)
            if time.time() < hold_until:
                return True

            # Check daily request budget
            limit = model.get("limits", {}).get("req_per_day")
            if limit and state["daily"].get(model_id, 0) >= limit:
                return True

            return False

    def record_success(self, model: dict) -> None:
        with _lock:
            state = self._sync()
            model_id = model["id"]
            state["daily"][model_id] = state["daily"].get(model_id, 0) + 1
            _save(state)

    def mark_rate_limited(self, model: dict, retry_after_seconds: int = 60) -> None:
        with _lock:
            state = self._sync()
            state["limited_until"][model["id"]] = time.time() + retry_after_seconds
            _save(state)

    def clear_hold(self, model_id: str) -> None:
        with _lock:
            state = self._sync()
            state["limited_until"].pop(model_id, None)
            _save(state)

    def status(self) -> dict:
        with _lock:
            state = self._sync()
            return {
                "date": state["date"],
                "daily_used": state["daily"],
                "held_until": {
                    k: datetime.fromtimestamp(v, tz=timezone.utc).isoformat()
                    for k, v in state["limited_until"].items()
                    if time.time() < v
                },
            }
