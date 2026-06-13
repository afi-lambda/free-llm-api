#!/usr/bin/env python3
"""
Refresh SWE-bench Verified scores in registry.json and re-tier models.

Scores are maintained here as the ground truth — update when the leaderboard
publishes new results (https://www.swebench.com/). The weekly cron runs this
before the smoke test so tiers reflect the latest published benchmarks.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

REGISTRY = Path(__file__).parent.parent / "registry.json"

# SWE-bench Verified scores — update from https://www.swebench.com/
# Key: model slug as it appears in our registry model_param field
# Value: fraction resolved (0.0–1.0)
KNOWN_SCORES: dict[str, float] = {
    # Google AI Studio
    "gemini-2.5-flash": 0.55,
    "gemini-3.5-flash": 0.52,
    "gemini-3-flash": 0.50,
    "gemini-2.5-flash-lite": 0.30,
    "gemini-3.1-flash-lite": 0.22,
    # OpenRouter (by model_param slug without :free)
    "qwen/qwen3-coder": 0.48,
    "deepseek/deepseek-v4-flash": 0.42,
    "moonshotai/kimi-k2.6": 0.40,
    "openai/gpt-oss-120b": 0.38,
    "qwen/qwen3-next-80b-a3b-instruct": 0.35,
    "minimax/minimax-m2.5": 0.32,
    "openai/gpt-oss-20b": 0.28,
    "meta-llama/llama-3.3-70b-instruct": 0.27,
    "nousresearch/hermes-3-llama-3.1-405b": 0.22,
    "meta-llama/llama-3.2-3b-instruct": 0.10,
    # Groq (by model_param)
    "meta-llama/llama-4-scout-17b-16e-instruct": 0.30,
    "llama-3.3-70b-versatile": 0.27,
    "qwen-qwen3-32b": 0.35,
    "llama-3.1-8b-instant": 0.15,
    # Cerebras
    "llama3.1-8b": 0.15,
    "gpt-oss-120b": 0.38,
}

# Last updated date — bump this when you refresh the scores above
SCORES_DATE = "2026-06-13"


def assign_tier(score: float) -> int:
    if score >= 0.45:
        return 1
    if score >= 0.25:
        return 2
    return 3


def apply(registry: dict) -> tuple[int, int]:
    """Apply KNOWN_SCORES to registry models. Returns (updated, unchanged)."""
    updated = unchanged = 0
    for model in registry["models"]:
        param = model.get("model_param", "").replace(":free", "")
        score = KNOWN_SCORES.get(param)
        if score is None:
            unchanged += 1
            continue
        old_tier = model.get("tier", 3)
        new_tier = assign_tier(score)
        model["swebench_score"] = score
        model["tier"] = new_tier
        if old_tier != new_tier:
            print(f"  Re-tiered: {model['id']}  T{old_tier} → T{new_tier}  (score={score:.0%})")
        updated += 1
    return updated, unchanged


def main() -> None:
    if not REGISTRY.exists():
        print("ERROR: registry.json not found. Run discovery/cheahjs_sync.py first.")
        return

    registry = json.loads(REGISTRY.read_text())
    print(f"Applying SWE-bench scores (dataset: {SCORES_DATE})...")
    updated, unchanged = apply(registry)

    registry["swebench_updated_at"] = datetime.now(timezone.utc).isoformat()
    registry["swebench_scores_date"] = SCORES_DATE

    # Re-sort by updated tiers
    registry["models"].sort(key=lambda m: (
        m.get("tier", 3),
        -m.get("swebench_score", 0.0),
        m.get("latency_ms") or 9999,
    ))

    REGISTRY.write_text(json.dumps(registry, indent=2))
    print(f"Done: {updated} scores applied, {unchanged} models unscored (will use smoke results).")

    tier_counts = {1: 0, 2: 0, 3: 0}
    for m in registry["models"]:
        tier_counts[m.get("tier", 3)] += 1
    print(f"  Tier 1: {tier_counts[1]}  Tier 2: {tier_counts[2]}  Tier 3: {tier_counts[3]}")


if __name__ == "__main__":
    main()
