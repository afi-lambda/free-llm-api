#!/usr/bin/env python3
"""
Async liveness probe for all models in registry.json.
Updates alive/latency_ms/last_checked fields in-place.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

REGISTRY = Path(__file__).parent.parent / "registry.json"
TIMEOUT = 15.0  # seconds per probe (thinking models need more)
PROBE_MSG = "Reply with just the number: 2+2"
# Thinking models need more tokens or they return an empty content field
PROBE_MAX_TOKENS = 64
# Max concurrent requests per provider (avoids 429 on shared quota pools)
PROVIDER_CONCURRENCY: dict[str, int] = {
    "openrouter": 4,   # 20 req/min shared — burst 4 at a time with gaps
    "google_ai_studio": 3,
    "groq": 5,
    "cerebras": 5,
}

# Extra headers required by some providers
PROVIDER_HEADERS: dict[str, dict] = {
    "openrouter": {
        "HTTP-Referer": "https://github.com/free-llm-pool",
        "X-Title": "free-llm-pool",
    },
}


def get_api_key(env_key: str) -> str | None:
    return os.environ.get(env_key) or os.environ.get(env_key.upper())


async def probe(client: httpx.AsyncClient, model: dict, sem: asyncio.Semaphore) -> dict:
    async with sem:
        api_key = get_api_key(model["env_key"])
        if not api_key:
            return {**model, "alive": None, "skip_reason": f"{model['env_key']} not set"}

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **PROVIDER_HEADERS.get(model["provider"], {}),
        }
        payload = {
            "model": model["model_param"],
            "messages": [{"role": "user", "content": PROBE_MSG}],
            "max_tokens": PROBE_MAX_TOKENS,
        }

        t0 = time.monotonic()
        try:
            resp = await client.post(
                f"{model['base_url']}/chat/completions",
                headers=headers,
                json=payload,
                timeout=TIMEOUT,
            )
            latency_ms = int((time.monotonic() - t0) * 1000)

            if resp.status_code == 200:
                data = resp.json()
                msg = data["choices"][0]["message"]
                content = (msg.get("content") or "").strip()
                alive = True
            else:
                alive = False
                content = f"HTTP {resp.status_code}: {resp.text[:120]}"
                latency_ms = None

        except httpx.TimeoutException:
            alive = False
            content = "timeout"
            latency_ms = None
        except Exception as e:
            alive = False
            content = str(e)[:120]
            latency_ms = None

        now = datetime.now(timezone.utc).isoformat()
        return {
            **model,
            "alive": alive,
            "latency_ms": latency_ms,
            "last_checked": now,
            "last_response": content if not alive else None,
        }


async def run_probes(models: list[dict]) -> list[dict]:
    # Group models by provider so each gets its own concurrency semaphore
    from collections import defaultdict
    by_provider: dict[str, list[dict]] = defaultdict(list)
    for m in models:
        by_provider[m["provider"]].append(m)

    async with httpx.AsyncClient() as client:
        tasks = []
        for provider, pmodels in by_provider.items():
            limit = PROVIDER_CONCURRENCY.get(provider, 4)
            sem = asyncio.Semaphore(limit)
            tasks += [probe(client, m, sem) for m in pmodels]
        results = await asyncio.gather(*tasks)
    return list(results)


def summarise(results: list[dict]) -> None:
    alive = [r for r in results if r.get("alive") is True]
    dead = [r for r in results if r.get("alive") is False]
    skipped = [r for r in results if r.get("alive") is None]

    print(f"\nLiveness: {len(alive)} alive  {len(dead)} dead  {len(skipped)} skipped (no key)")

    if alive:
        print("\n  ALIVE (by tier):")
        for r in sorted(alive, key=lambda x: (x.get("tier", 3), x.get("latency_ms") or 9999)):
            lat = f"{r['latency_ms']}ms" if r.get("latency_ms") else "?"
            print(f"    T{r.get('tier','?')}  {r['id']:55s}  {lat}")

    if dead:
        print("\n  DEAD:")
        for r in sorted(dead, key=lambda x: x.get("tier", 3)):
            reason = r.get("last_response", "")
            print(f"    T{r.get('tier','?')}  {r['id']:55s}  {reason[:60]}")

    if skipped:
        keys_missing = sorted({r["env_key"] for r in skipped})
        print(f"\n  Skipped (missing keys): {', '.join(keys_missing)}")


def main() -> None:
    if not REGISTRY.exists():
        print("ERROR: registry.json not found. Run discovery/cheahjs_sync.py first.", file=sys.stderr)
        sys.exit(1)

    registry = json.loads(REGISTRY.read_text())
    models = registry["models"]
    print(f"Probing {len(models)} models...")

    results = asyncio.run(run_probes(models))

    # Write back to registry
    registry["models"] = results
    registry["liveness_checked_at"] = datetime.now(timezone.utc).isoformat()
    REGISTRY.write_text(json.dumps(registry, indent=2))

    summarise(results)


if __name__ == "__main__":
    main()
