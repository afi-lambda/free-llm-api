# Free LLM API Pool — Implementation Plan

## Goal

A self-maintaining pool of free LLM APIs, ranked by coding quality (SWE-bench),
with automatic discovery, health checks, and fallback routing — ready before 6am.

---

## Reference

- **cheahjs/free-llm-api-resources** — community-maintained list of free providers with
  limits and model names. Their `src/pull_available_models.py` scrapes live data from
  each provider. We consume their output rather than scraping ourselves.
- **SWE-bench Verified leaderboard** — ground truth for coding quality ranking.

---

## Architecture

```
discovery/          # Phase 1 — find & classify models
  cheahjs_sync.py   #   pull latest model list from cheahjs repo (daily)
  swebench.py       #   fetch/cache SWE-bench scores (weekly)
  registry.json     #   output: {provider, model, tier, limits, last_seen}

health/             # Phase 2 — liveness & quality
  liveness.py       #   async HTTP probe each provider (5s timeout)
  smoke.py          #   run 10-problem coding benchmark against live models
  results.json      #   output: {model, alive, latency_ms, smoke_score, checked_at}

router/             # Phase 3 — request routing
  pool.py           #   load registry + results → ordered candidate list
  client.py         #   send request, rotate on 429/503, track rate limits
  rate_tracker.py   #   per-provider token/request budget, window reset

cron/               # Phase 4 — scheduling
  daily.sh          #   runs at 05:45 — cheahjs sync + liveness probe
  weekly.sh         #   runs Monday 05:45 — smoke test + SWE-bench refresh
```

---

## Phase 1 — Discovery & Classification

### 1.1 cheahjs sync (daily, 05:45)

```python
# discovery/cheahjs_sync.py
# Fetch https://raw.githubusercontent.com/cheahjs/free-llm-api-resources/main/README.md
# Parse provider blocks → extract model IDs, rate limits
# Diff against registry.json → log additions / removals
# Write updated registry.json
```

Key providers to track (from cheahjs README as of 2026-06-13):
| Provider | Free models | Key limits |
|----------|-------------|------------|
| OpenRouter | 25+ `:free` models | 20 req/min, 50 req/day (1000 with $10 topup) |
| Google AI Studio | Gemini 2.5 Flash, Gemma 3 | 20 req/day (Gemini), 14400/day (Gemma) |
| Mistral Codestral | Codestral | 30 req/min, 2000 req/day |
| Cerebras | Llama 3.3 70B, Qwen | high speed |
| Groq | Llama 4, Gemma | per-model limits |
| NVIDIA NIM | Open models | 40 req/min |
| GitHub Models | GPT-4o-mini, etc. | per-account |

### 1.2 Tier assignment

Tiers based on SWE-bench score + availability:

```
Tier 1 (>50% SWE-bench):  Gemini 2.5 Flash (Google AI Studio)
Tier 2 (30–50%):          DeepSeek V4 Flash (OpenRouter), Qwen3-Coder (OpenRouter),
                           Codestral (Mistral), Llama 3.3 70B (Groq/Cerebras)
Tier 3 (<30%, fallback):  Everything else in the free pool
```

registry.json schema:
```json
{
  "models": [
    {
      "id": "google/gemini-2.5-flash",
      "provider": "google_ai_studio",
      "tier": 1,
      "swebench_score": 0.63,
      "limits": {"req_per_day": 20, "req_per_min": 5},
      "openai_compat": true,
      "last_seen": "2026-06-13",
      "alive": true
    }
  ]
}
```

---

## Phase 2 — Health Checks

### 2.1 Liveness probe (daily, runs after sync)

- Send `"What is 2+2?"` to each model, 5s timeout
- Record: HTTP status, latency_ms, non-empty response
- Mark dead if 3 consecutive failures

### 2.2 Smoke test (weekly, Monday 05:45)

10 fixed coding problems from HumanEval (problems 0, 1, 5, 11, 17, 26, 32, 40, 62, 77).
Score = correct/10. Update tier if score drops >2 points from baseline.

```python
# health/smoke.py
PROBLEMS = load("benchmark/humaneval_10.json")  # fixed, never changes

async def run_smoke(model_id, client):
    scores = [eval_problem(p, await client.complete(p.prompt)) for p in PROBLEMS]
    return sum(scores) / len(scores)
```

---

## Phase 3 — Router

### 3.1 Candidate selection

```python
# router/pool.py
def get_candidates():
    # Load registry.json + results.json
    # Filter: alive=True, last_seen within 2 days
    # Sort: tier ASC, swebench_score DESC, latency_ms ASC
    # Return ordered list
```

### 3.2 Request routing with fallback

```python
# router/client.py
async def complete(prompt, max_retries=len(candidates)):
    for model in get_candidates():
        if rate_tracker.is_exhausted(model):
            continue
        try:
            resp = await call(model, prompt, timeout=30)
            rate_tracker.record(model, resp.headers)
            return resp
        except (RateLimitError, ServiceUnavailable):
            rate_tracker.mark_limited(model)
            continue
    raise AllProvidersExhausted()
```

### 3.3 Rate limit tracking

- Read `x-ratelimit-remaining-requests` and `x-ratelimit-reset` headers
- Track per-provider daily budget (reset at midnight UTC)
- Providers without standard headers: track manually with counter + window

---

## Phase 4 — Scheduling (cron)

```bash
# cron/daily.sh — runs at 05:45 every day
cd /home/alain/free-llm-api
source venv/bin/activate
python3 discovery/cheahjs_sync.py   # diff free model list
python3 health/liveness.py          # probe all live models
python3 -m notify "Pool ready: $(python3 router/pool.py --count) models available"
```

```bash
# cron/weekly.sh — runs Monday 05:45
cd /home/alain/free-llm-api
source venv/bin/activate
python3 discovery/swebench.py       # refresh SWE-bench scores
python3 health/smoke.py             # re-score all Tier 1+2 models
python3 discovery/cheahjs_sync.py   # re-tier if scores changed
```

Crontab entries:
```
45 5 * * *   /home/alain/free-llm-api/cron/daily.sh >> /tmp/llm-pool-daily.log 2>&1
45 5 * * 1   /home/alain/free-llm-api/cron/weekly.sh >> /tmp/llm-pool-weekly.log 2>&1
```

---

## Robustness targets

| Metric | Target |
|--------|--------|
| Tier 1 models always available | ≥1 |
| Total fallback candidates | ≥5 across ≥3 providers |
| Time to detect provider outage | <24h (liveness) |
| Time to detect quality regression | <7 days (smoke) |
| Time to detect model removal | <24h (cheahjs diff) |

---

## Implementation order

1. `discovery/cheahjs_sync.py` + `registry.json` schema — foundation everything else reads
2. `health/liveness.py` — know what's alive before routing
3. `router/client.py` + `router/rate_tracker.py` — usable pool
4. `cron/daily.sh` — automation
5. `benchmark/humaneval_10.json` + `health/smoke.py` — quality gate
6. `discovery/swebench.py` — close the tier feedback loop
7. `cron/weekly.sh` — full automation

---

## API key setup needed

```bash
# .env (never commit)
GOOGLE_AI_STUDIO_KEY=...    # aistudio.google.com → Get API key
OPENROUTER_API_KEY=...      # openrouter.ai → Keys
MISTRAL_API_KEY=...         # console.mistral.ai → API keys
GROQ_API_KEY=...            # console.groq.com → API keys
GITHUB_TOKEN=...            # github.com → Settings → Developer → Tokens
# NVIDIA NIM: phone verification required — optional
```

---

## Open questions before starting

1. Do you have API keys for these providers already?
2. Should the router expose an OpenAI-compatible endpoint (`/v1/chat/completions`)
   so other tools (hermes-agent, kb enrichment) can point at it transparently?
3. For smoke tests: run against all Tier 1+2 models weekly, or only the top 3
   to stay within free daily quotas?
