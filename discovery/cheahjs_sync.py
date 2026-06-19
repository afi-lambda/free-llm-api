#!/usr/bin/env python3
"""
Sync free LLM model list from cheahjs/free-llm-api-resources.
Diffs against registry.json and logs additions/removals.
"""

import base64
import json
import os
import re
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import httpx

REGISTRY = Path(__file__).parent.parent / "registry.json"

# Seeded SWE-bench Verified scores — updated weekly by swebench.py
SWEBENCH_SCORES: dict[str, float] = {
    # Google AI Studio
    "gemini-2.5-flash": 0.55,
    "gemini-2.5-flash-lite": 0.30,
    "gemini-3-flash": 0.50,
    "gemini-3.5-flash": 0.52,
    "gemma-3-27b-it": 0.20,
    # OpenRouter (by model_param slug)
    "qwen/qwen3-coder": 0.48,
    "deepseek/deepseek-v4-flash": 0.42,
    "moonshotai/kimi-k2.6": 0.40,
    "openai/gpt-oss-120b": 0.38,
    "qwen/qwen3-next-80b-a3b-instruct": 0.35,
    "minimax/minimax-m2.5": 0.32,
    "openai/gpt-oss-20b": 0.28,
    "meta-llama/llama-3.3-70b-instruct": 0.27,
    "nousresearch/hermes-3-llama-3.1-405b": 0.22,
    # Groq (by model_param)
    "meta-llama/llama-4-scout-17b-16e-instruct": 0.30,
    "llama-3.3-70b-versatile": 0.27,
    "qwen/qwen3-32b": 0.35,
    "llama-3.1-8b-instant": 0.15,
    # Ollama Cloud (by model_param) — reuse scores from equivalent models above
    "gpt-oss:120b": 0.38,
    "qwen3-coder:480b": 0.48,
}


def tier(model_id: str) -> int:
    score = SWEBENCH_SCORES.get(model_id, 0.0)
    if score >= 0.45:
        return 1
    if score >= 0.25:
        return 2
    return 3


def fetch_readme() -> str:
    result = subprocess.run(
        ["gh", "api", "repos/cheahjs/free-llm-api-resources/readme", "--jq", ".content"],
        capture_output=True, text=True, check=True,
    )
    return base64.b64decode(result.stdout.strip()).decode()


def split_sections(readme: str) -> dict[str, str]:
    """Return {provider_name: section_text} for each ### heading."""
    sections: dict[str, str] = {}
    current_name = ""
    current_lines: list[str] = []
    for line in readme.splitlines():
        if line.startswith("### "):
            if current_name:
                sections[current_name] = "\n".join(current_lines)
            m = re.match(r"### \[([^\]]+)\]", line)
            current_name = m.group(1) if m else line[4:].strip()
            current_lines = [line]
        elif current_name:
            current_lines.append(line)
    if current_name:
        sections[current_name] = "\n".join(current_lines)
    return sections


def parse_limits(text: str) -> dict:
    limits: dict = {}
    if m := re.search(r"([\d,]+) requests/day", text):
        limits["req_per_day"] = int(m.group(1).replace(",", ""))
    if m := re.search(r"([\d,]+) requests/minute", text):
        limits["req_per_min"] = int(m.group(1).replace(",", ""))
    if m := re.search(r"([\d,]+) tokens/minute", text):
        limits["tokens_per_min"] = int(m.group(1).replace(",", ""))
    return limits


def fetch_openrouter_limits(api_key: str) -> dict:
    """Free model variants (IDs ending in :free) get 1000 req/day once the
    account has purchased >=10 credits (is_free_tier == False), else 50.
    req/min is a flat 20 regardless of tier. The `rate_limit` field on this
    endpoint is marked deprecated by OpenRouter, so it's ignored here.
    https://openrouter.ai/docs/api/reference/limits
    """
    try:
        resp = httpx.get(
            "https://openrouter.ai/api/v1/key",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        resp.raise_for_status()
        is_free_tier = resp.json()["data"]["is_free_tier"]
    except (httpx.HTTPError, KeyError, ValueError) as e:
        print(f"  Failed to fetch OpenRouter key status ({e}); assuming free-tier limits", file=sys.stderr)
        is_free_tier = True
    return {"req_per_day": 50 if is_free_tier else 1000, "req_per_min": 20}

# Non-chat model substrings to skip across all providers
SKIP_MODEL_KEYWORDS = {
    "whisper", "tts", "guard", "safeguard", "orpheus",
    "robotics", "llava", "lyria", "clip", "safety",
    "ultra-550b",  # nemotron-3-ultra returns 200 with no choices key
}

def fetch_openrouter_free_models() -> list[dict]:
    """Query OpenRouter /models API — pricing.prompt == '0' means free."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("  OPENROUTER_API_KEY not set — skipping OpenRouter", file=sys.stderr)
        return []
    limits = fetch_openrouter_limits(api_key)
    resp = httpx.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=15,
    )
    resp.raise_for_status()
    models = []
    for m in resp.json()["data"]:
        slug = m["id"]  # e.g. "qwen/qwen3-coder:free"
        # Only include free models that end with :free or have prompt price == 0
        pricing = m.get("pricing", {})
        if str(pricing.get("prompt", "1")) != "0":
            continue
        # Skip non-chat models
        if any(kw in slug.lower() for kw in SKIP_MODEL_KEYWORDS):
            continue
        # Skip router/meta endpoints
        if "/" not in slug or slug.startswith("openrouter/"):
            continue
        model_id = slug.replace(":free", "")
        models.append({
            "id": f"openrouter/{model_id}",
            "display_name": m.get("name", model_id.split("/")[-1]),
            "provider": "openrouter",
            "openai_compat": True,
            "base_url": "https://openrouter.ai/api/v1",
            "model_param": slug,
            "limits": limits,
            "env_key": "OPENROUTER_API_KEY",
            "context_length": m.get("context_length"),
        })
    return models


# Maps cheahjs display name → (model_id, base_url, env_key) for table-based providers
GOOGLE_NAME_MAP: dict[str, str] = {
    # Only Gemini models work on the OpenAI-compat endpoint (/v1beta/openai/)
    # Gemma models return 404 there — they need the native generateContent endpoint
    "Gemini 3.5 Flash": "gemini-3.5-flash",
    # "Gemini 3 Flash" returns 404 on OpenAI-compat endpoint — skip until confirmed live
    "Gemini 3.1 Flash-Lite": "gemini-3.1-flash-lite",
    "Gemini 2.5 Flash": "gemini-2.5-flash",
    "Gemini 2.5 Flash-Lite": "gemini-2.5-flash-lite",
}

# Non-chat models to skip (speech, safety, vision-only, etc.) — kept for table providers
_TABLE_SKIP_KEYWORDS = SKIP_MODEL_KEYWORDS

GROQ_NAME_MAP: dict[str, str] = {
    "Llama 3.1 8B": "llama-3.1-8b-instant",
    "Llama 3.3 70B": "llama-3.3-70b-versatile",
    "Llama 4 Scout Instruct": "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen/qwen3-32b": "qwen/qwen3-32b",
    "openai/gpt-oss-120b": "openai/gpt-oss-120b",
    "openai/gpt-oss-20b": "openai/gpt-oss-20b",
    "groq/compound": "groq/compound",
    "groq/compound-mini": "groq/compound-mini",
}

CEREBRAS_NAME_MAP: dict[str, str] = {
    "gpt-oss-120b": "gpt-oss-120b",
}

# Cerebras has discontinued llama3.1-8b (404 on /v1/chat/completions, no
# longer listed in /v1/models) but cheahjs's README hasn't been updated to
# drop it, so it's intentionally left out of CEREBRAS_NAME_MAP above.

# zai-glm-4.7 is live on Cerebras's /v1/models but not yet listed in
# cheahjs's README table, so it's curated here. Limits reused from
# gpt-oss-120b's row in that table — Cerebras applies uniform free-tier
# limits across all their models.
CEREBRAS_EXTRA_MODELS: dict[str, dict] = {
    "zai-glm-4.7": {"req_per_day": 14400, "req_per_min": 30, "tokens_per_min": 60000},
}


def fetch_cerebras_extra_models() -> list[dict]:
    if not os.environ.get("CEREBRAS_API_KEY"):
        print("  CEREBRAS_API_KEY not set — skipping curated Cerebras extras", file=sys.stderr)
        return []
    return [
        {
            "id": f"cerebras/{model_param}",
            "display_name": model_param,
            "provider": "cerebras",
            "openai_compat": True,
            "base_url": "https://api.cerebras.ai/v1",
            "model_param": model_param,
            "limits": limits,
            "env_key": "CEREBRAS_API_KEY",
        }
        for model_param, limits in CEREBRAS_EXTRA_MODELS.items()
    ]

# Ollama Cloud — no cheahjs section and no public discovery API for free
# cloud-tagged models, so this list is manually curated from
# https://ollama.com/search?c=cloud. Free tier is GPU-time-based
# (session resets every 5h, weekly reset every 7 days), not request-count
# based, so no req_per_day/req_per_min limits are set here.
OLLAMA_CLOUD_MODELS: list[str] = [
    "gpt-oss:120b",
    "qwen3-coder:480b",
    "gemma4:31b",
]


def fetch_ollama_cloud_models() -> list[dict]:
    if not os.environ.get("OLLAMA_API_KEY"):
        print("  OLLAMA_API_KEY not set — skipping Ollama Cloud", file=sys.stderr)
        return []
    return [
        {
            "id": f"ollama/{model_param}",
            "display_name": model_param,
            "provider": "ollama",
            "openai_compat": True,
            "base_url": "https://ollama.com/v1",
            "model_param": model_param,
            "limits": {},
            "env_key": "OLLAMA_API_KEY",
        }
        for model_param in OLLAMA_CLOUD_MODELS
    ]


def parse_html_table_provider(
    section: str,
    provider: str,
    base_url: str,
    env_key: str,
    name_map: dict[str, str],
) -> list[dict]:
    models = []
    for m in re.finditer(r"<tr><td>([^<]+)</td><td>([^<]*(?:<br>[^<]*)*)</td></tr>", section):
        display_name, limits_text = m.group(1).strip(), m.group(2)
        # Skip non-chat models
        name_lower = display_name.lower()
        if any(kw in name_lower for kw in _TABLE_SKIP_KEYWORDS):
            continue
        model_id = name_map.get(display_name)
        if not model_id:
            # Try table cells where model ID is in the name itself (e.g. groq table has raw IDs)
            if "/" in display_name or re.match(r"[a-z]", display_name):
                model_id = display_name
            else:
                continue
        limits = parse_limits(limits_text.replace("<br>", "\n"))
        # Registry ID: strip leading provider-name prefix from model_id to avoid doubling
        # e.g. model_id="groq/compound" → id="groq/compound" not "groq/groq/compound"
        id_suffix = model_id[len(provider) + 1:] if model_id.startswith(f"{provider}/") else model_id
        models.append({
            "id": f"{provider}/{id_suffix}",
            "display_name": display_name,
            "provider": provider,
            "openai_compat": True,
            "base_url": base_url,
            "model_param": model_id,
            "limits": limits,
            "env_key": env_key,
        })
    return models


def parse_models(readme: str) -> list[dict]:
    sections = split_sections(readme)
    models: list[dict] = []

    # OpenRouter: use their API directly (ground truth), not cheahjs README
    print("  Fetching OpenRouter free model list from API...")
    models += fetch_openrouter_free_models()

    if s := sections.get("Google AI Studio"):
        models += parse_html_table_provider(
            s, "google_ai_studio",
            "https://generativelanguage.googleapis.com/v1beta/openai",
            "GEMINI_API_KEY", GOOGLE_NAME_MAP,
        )

    if s := sections.get("Groq"):
        models += parse_html_table_provider(
            s, "groq",
            "https://api.groq.com/openai/v1",
            "GROQ_API_KEY", GROQ_NAME_MAP,
        )

    if s := sections.get("Cerebras"):
        models += parse_html_table_provider(
            s, "cerebras",
            "https://api.cerebras.ai/v1",
            "CEREBRAS_API_KEY", CEREBRAS_NAME_MAP,
        )

    print("  Adding curated Cerebras extras...")
    models += fetch_cerebras_extra_models()

    print("  Adding curated Ollama Cloud model list...")
    models += fetch_ollama_cloud_models()

    # Annotate with tier and SWE-bench score
    for m in models:
        core_id = m["id"].replace(f"{m['provider']}/", "", 1)
        m["swebench_score"] = SWEBENCH_SCORES.get(core_id, SWEBENCH_SCORES.get(m["model_param"], 0.0))
        m["tier"] = tier(core_id) if core_id in SWEBENCH_SCORES else tier(m["model_param"])

    return models


def load_registry() -> dict:
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text())
    return {"updated_at": None, "models": []}


def save_registry(registry: dict) -> None:
    registry["updated_at"] = datetime.now(timezone.utc).isoformat()
    for m in registry["models"]:
        m.setdefault("alive", None)
        m.setdefault("last_checked", None)
        m.setdefault("last_seen", str(date.today()))
    REGISTRY.write_text(json.dumps(registry, indent=2))


def sync() -> None:
    print("Fetching cheahjs/free-llm-api-resources README...")
    readme = fetch_readme()

    print("Parsing models...")
    fresh = {m["id"]: m for m in parse_models(readme)}

    old_registry = load_registry()
    old = {m["id"]: m for m in old_registry.get("models", [])}

    added = [k for k in fresh if k not in old]
    removed = [k for k in old if k not in fresh]
    today = str(date.today())

    if added:
        print(f"  + {len(added)} new models: {', '.join(added[:5])}{'...' if len(added) > 5 else ''}")
    if removed:
        print(f"  - {len(removed)} removed models: {', '.join(removed[:5])}{'...' if len(removed) > 5 else ''}")
    if not added and not removed:
        print("  No changes in model list.")

    # Merge: preserve liveness/alive state from old entries
    merged: list[dict] = []
    for model_id, model in fresh.items():
        if model_id in old:
            model["alive"] = old[model_id].get("alive")
            model["last_checked"] = old[model_id].get("last_checked")
        model["last_seen"] = today
        merged.append(model)

    # Keep removed models for 2 days (grace period for transient outages)
    for model_id in removed:
        old_entry = old[model_id]
        last_seen = old_entry.get("last_seen", today)
        days_missing = (date.today() - date.fromisoformat(last_seen)).days
        if days_missing < 2:
            old_entry["tier"] = max(old_entry.get("tier", 3), 3)  # demote to tier 3
            merged.append(old_entry)
        else:
            print(f"  Dropping {model_id} (missing {days_missing} days)")

    # Sort by tier, then swebench_score desc
    merged.sort(key=lambda m: (m.get("tier", 3), -m.get("swebench_score", 0.0)))

    save_registry({"models": merged})
    print(f"Registry written: {len(merged)} models across "
          f"{len(set(m['provider'] for m in merged))} providers.")
    print(f"  Tier 1: {sum(1 for m in merged if m.get('tier') == 1)}"
          f"  Tier 2: {sum(1 for m in merged if m.get('tier') == 2)}"
          f"  Tier 3: {sum(1 for m in merged if m.get('tier') == 3)}")


if __name__ == "__main__":
    try:
        sync()
    except subprocess.CalledProcessError as e:
        print(f"ERROR: gh CLI failed: {e.stderr}", file=sys.stderr)
        sys.exit(1)
