#!/usr/bin/env python3
"""
Weekly smoke test: run 10 HumanEval problems against Tier 1+2 models.
Updates swebench_score_observed in registry.json and re-tiers if regression detected.
"""

import asyncio
import json
import os
import re
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import httpx

REGISTRY = Path(__file__).parent.parent / "registry.json"
BENCHMARK = Path(__file__).parent.parent / "benchmark" / "humaneval_10.json"
TIMEOUT = 45.0
# Thinking models consume tokens internally before producing output
THINKING_MODELS = {"gemini-2.5", "gemini-3", "qwen3", "deepseek-r1"}
DEFAULT_MAX_TOKENS = 512
THINKING_MAX_TOKENS = 2048

PROVIDER_HEADERS: dict[str, dict] = {
    "openrouter": {
        "HTTP-Referer": "https://github.com/free-llm-pool",
        "X-Title": "free-llm-pool",
    },
}

SYSTEM_PROMPT = (
    "You are a Python coding assistant. "
    "When given a function signature and docstring, complete the function. "
    "Output ONLY the complete Python function — no explanation, no markdown fences."
)


def get_api_key(env_key: str) -> str | None:
    return os.environ.get(env_key) or os.environ.get(env_key.upper())


def extract_code(text: str, entry_point: str) -> str:
    """Strip markdown fences and extract the function body."""
    text = re.sub(r"```(?:python)?\n?", "", text).replace("```", "").strip()
    # Keep only lines starting from the function definition
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith("def ")), 0)
    return "\n".join(lines[start:])


def run_tests(code: str, problem: dict) -> tuple[int, int]:
    """Execute generated code against test assertions. Returns (passed, total)."""
    namespace: dict = {}
    passed = 0
    try:
        exec(textwrap.dedent(problem["prompt"]) + "\n" + code, namespace)
    except Exception:
        return 0, len(problem["tests"])

    for assertion in problem["tests"]:
        try:
            exec(assertion, namespace)
            passed += 1
        except Exception:
            pass
    return passed, len(problem["tests"])


async def score_model(
    client: httpx.AsyncClient,
    model: dict,
    problems: list[dict],
) -> tuple[str, float]:
    api_key = get_api_key(model["env_key"])
    if not api_key:
        return model["id"], -1.0

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        **PROVIDER_HEADERS.get(model["provider"], {}),
    }

    total_passed = 0
    total_tests = 0

    param = model["model_param"].lower()
    is_thinking = any(kw in param for kw in THINKING_MODELS)
    max_tok = THINKING_MAX_TOKENS if is_thinking else DEFAULT_MAX_TOKENS

    for problem in problems:
        payload = {
            "model": model["model_param"],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem["prompt"]},
            ],
            "max_tokens": max_tok,
            "temperature": 0.0,
        }
        try:
            resp = await client.post(
                f"{model['base_url']}/chat/completions",
                headers=headers,
                json=payload,
                timeout=TIMEOUT,
            )
            if resp.status_code == 429:
                return model["id"], -1.0  # quota exhausted — skip, don't score 0
            if resp.status_code != 200:
                total_tests += len(problem["tests"])
                continue
            raw = resp.json()["choices"][0]["message"].get("content") or ""
            code = extract_code(raw, problem["entry_point"])
            p, t = run_tests(code, problem)
            total_passed += p
            total_tests += t
        except Exception:
            total_tests += len(problem["tests"])

    score = total_passed / total_tests if total_tests > 0 else 0.0
    return model["id"], score


async def run_smoke(models: list[dict], problems: list[dict]) -> dict[str, float]:
    async with httpx.AsyncClient() as client:
        tasks = [score_model(client, m, problems) for m in models]
        results = await asyncio.gather(*tasks)
    return dict(results)


def assign_tier(score: float) -> int:
    if score >= 0.45:
        return 1
    if score >= 0.25:
        return 2
    return 3


def main() -> None:
    if not REGISTRY.exists():
        print("ERROR: registry.json not found.", file=sys.stderr)
        sys.exit(1)
    if not BENCHMARK.exists():
        print("ERROR: benchmark/humaneval_10.json not found.", file=sys.stderr)
        sys.exit(1)

    registry = json.loads(REGISTRY.read_text())
    problems = json.loads(BENCHMARK.read_text())

    # Smoke only Tier 1 + 2 alive models (+ one rotating Tier 3 for coverage)
    candidates = [m for m in registry["models"] if m.get("tier", 3) <= 2 and m.get("alive") is not False]

    # Rotating Tier 3 pick: week number mod count
    tier3 = [m for m in registry["models"] if m.get("tier", 3) == 3 and m.get("alive") is not False]
    if tier3:
        week = datetime.now(timezone.utc).isocalendar()[1]
        candidates.append(tier3[week % len(tier3)])

    print(f"Smoke testing {len(candidates)} models against {len(problems)} problems...")

    scores = asyncio.run(run_smoke(candidates, problems))

    # Update registry
    regressions = []
    for model in registry["models"]:
        if model["id"] not in scores:
            continue
        new_score = scores[model["id"]]
        if new_score < 0:
            continue  # skipped (no key or quota exhausted)
        old_score = model.get("swebench_score", 0.0)
        new_tier = assign_tier(new_score)
        model["smoke_score"] = round(new_score, 3)
        model["smoke_checked_at"] = datetime.now(timezone.utc).isoformat()

        # Detect regression: score dropped >0.15 or tier changed
        if old_score > 0 and (old_score - new_score > 0.15 or new_tier > model.get("tier", 3)):
            regressions.append((model["id"], old_score, new_score))
            model["tier"] = new_tier

    REGISTRY.write_text(json.dumps(registry, indent=2))

    # Report
    print(f"\n{'Model':<55}  {'Smoke':>6}  {'Seed':>6}  Tier")
    print("-" * 75)
    for model in sorted(candidates, key=lambda m: -scores.get(m["id"], -1)):
        s = scores.get(model["id"], -1)
        seed = model.get("swebench_score", 0.0)
        smoke_str = f"{s:.0%}" if s >= 0 else "quota"
        print(f"  {model['id']:<53}  {smoke_str:>6}  {seed:.0%}  T{model.get('tier','?')}")

    if regressions:
        print("\n  REGRESSIONS DETECTED:")
        for mid, old, new in regressions:
            print(f"    {mid}: {old:.0%} → {new:.0%}")
    else:
        print("\n  No regressions.")


if __name__ == "__main__":
    main()
