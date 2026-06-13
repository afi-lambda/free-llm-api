"""
Routing client: tries candidates in priority order, rotates on 429/503.
Supports both streaming and non-streaming responses.
"""

import asyncio
import os
from typing import AsyncIterator

import httpx

from router.pool import candidates
from router.rate_tracker import RateTracker

_tracker = RateTracker()

PROVIDER_HEADERS: dict[str, dict] = {
    "openrouter": {
        "HTTP-Referer": "https://github.com/free-llm-pool",
        "X-Title": "free-llm-pool",
    },
}

RETRY_STATUS = {429, 503, 502, 504}
REQUEST_TIMEOUT = 60.0


def _headers(model: dict, api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        **PROVIDER_HEADERS.get(model["provider"], {}),
    }


def _api_key(model: dict) -> str | None:
    return os.environ.get(model["env_key"])


async def complete(
    messages: list[dict],
    *,
    model_hint: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    stream: bool = False,
    extra_params: dict | None = None,
) -> dict | AsyncIterator[bytes]:
    """
    Send a chat request to the best available model.
    Returns the full response dict (non-streaming) or an async byte iterator (streaming).
    Raises AllProvidersExhausted if every candidate fails.
    """
    pool = candidates()
    if not pool:
        raise AllProvidersExhausted("Registry empty or no keys configured")

    errors: list[str] = []

    for model in pool:
        if _tracker.is_exhausted(model):
            errors.append(f"{model['id']}: daily quota exhausted")
            continue

        api_key = _api_key(model)
        if not api_key:
            continue

        # If caller requested a specific model family, skip others unless we've run out
        if model_hint and model_hint.lower() not in model["id"].lower():
            if len(errors) < len(pool) // 2:
                continue

        payload = {
            "model": model["model_param"],
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **(extra_params or {}),
        }
        if stream:
            payload["stream"] = True

        try:
            if stream:
                return _stream(model, api_key, payload)
            else:
                resp = await _post(model, api_key, payload)
                _tracker.record_success(model)
                return resp
        except RateLimited as e:
            retry_after = e.retry_after
            _tracker.mark_rate_limited(model, retry_after)
            errors.append(f"{model['id']}: rate limited ({retry_after}s hold)")
            continue
        except ProviderError as e:
            errors.append(f"{model['id']}: {e}")
            continue

    raise AllProvidersExhausted("\n".join(errors))


async def _post(model: dict, api_key: str, payload: dict) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{model['base_url']}/chat/completions",
            headers=_headers(model, api_key),
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
    if resp.status_code in RETRY_STATUS:
        retry_after = int(resp.headers.get("retry-after", 60))
        raise RateLimited(retry_after)
    if resp.status_code != 200:
        raise ProviderError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    return resp.json()


async def _stream(model: dict, api_key: str, payload: dict) -> AsyncIterator[bytes]:
    """Yield raw SSE bytes from the upstream provider."""
    client = httpx.AsyncClient()
    req = client.stream(
        "POST",
        f"{model['base_url']}/chat/completions",
        headers=_headers(model, api_key),
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    async with req as resp:
        if resp.status_code in RETRY_STATUS:
            retry_after = int(resp.headers.get("retry-after", 60))
            _tracker.mark_rate_limited(model, retry_after)
            raise RateLimited(retry_after)
        if resp.status_code != 200:
            body = await resp.aread()
            raise ProviderError(f"HTTP {resp.status_code}: {body[:200]}")
        _tracker.record_success(model)
        async for chunk in resp.aiter_bytes():
            yield chunk
    await client.aclose()


# Exceptions

class AllProvidersExhausted(Exception):
    pass

class RateLimited(Exception):
    def __init__(self, retry_after: int = 60):
        self.retry_after = retry_after
        super().__init__(f"Rate limited, retry after {retry_after}s")

class ProviderError(Exception):
    pass
