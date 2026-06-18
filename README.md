# free-llm-api

Self-maintaining pool of free LLM APIs, ranked by coding quality, with automatic discovery, health checks, fallback routing, and a weekly HumanEval benchmark.

## Architecture

```
discovery/          Model discovery & tier assignment
  cheahjs_sync.py     Daily: pull free model list from cheahjs/free-llm-api-resources,
                      plus curated entries for providers/models not yet listed there
                      (Ollama Cloud, Cerebras zai-glm-4.7)
  swebench.py         Weekly: refresh SWE-bench Lite scores for tier assignment

health/             Liveness & quality checks
  liveness.py         Daily: async HTTP probe every model (5 s timeout)
  smoke.py            Weekly: run 10 HumanEval problems against Tier 1+2 models
  smoke_load_test.py  Manual: ramp Hermes-backed smoke concurrency 1→2→4→6→8→10,
                      sampling RAM/swap/load while it runs (capacity check, not CI)

router/             OpenAI-compatible proxy server
  pool.py             Ordered candidate list from registry.json
  client.py           Routing with fallback on 429/503; reads x-ratelimit headers
  rate_tracker.py     Per-model budget tracking, persisted to .rate_state.json

benchmark/
  humaneval_10.json   10 curated HumanEval problems (IDs: 0,5,11,26,32,40,62,77,83,107)

cron/               Scheduling scripts
  start-server.sh     Launch the proxy server
  daily.sh            05:45 UTC — cheahjs sync + liveness probe
  weekly.sh           05:48 UTC Monday — SWE-bench refresh + smoke test
```

## Tier system

Models are ranked by SWE-bench Lite score and validated weekly by the HumanEval smoke test:

| Tier | Smoke score | Role |
|------|-------------|------|
| T1   | ≥ 45%       | Primary — always tried first |
| T2   | 25–44%      | Secondary fallback |
| T3   | < 25%       | Last resort; one rotates into weekly smoke |

Smoke scores override tier on regression (score drops > 15 pp or tier demotes).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

API keys go in `~/.profile` (sourced by cron scripts):

```bash
export GEMINI_API_KEY=...        # aistudio.google.com
export OPENROUTER_API_KEY=...    # openrouter.ai/keys
export GROQ_API_KEY=...          # console.groq.com
export CEREBRAS_API_KEY=...      # inference.cerebras.ai
export OLLAMA_API_KEY=...        # ollama.com/settings/keys
```

## Running the server

```bash
./cron/start-server.sh           # port 8000
FREE_LLM_PORT=8001 ./cron/start-server.sh
```

Point any OpenAI-compatible tool at it:

```bash
OPENAI_BASE_URL=http://localhost:8000
OPENAI_API_KEY=free-pool         # ignored — real keys injected by router
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/chat/completions`, `/v1/chat/completions` | OpenAI Chat API |
| POST | `/messages`, `/v1/messages` | Anthropic Messages API |
| GET  | `/v1/models` | List pool models |
| GET  | `/health` | Pool status + rate tracker state |

### Example

```bash
curl http://localhost:8000/chat/completions \
  -H "Authorization: Bearer free-pool" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "hello"}]}'
```

## Smoke test

```bash
# Default: T1 + T2 alive models + one rotating T3
python3 health/smoke.py

# Only models without a score yet (bootstrap new registry entries)
python3 health/smoke.py --unscored
```

On 429 responses the script logs the full error body to stderr and records `quota_retry_after_seconds` / `quota_reset_at` in `registry.json`. The report shows `quota/Ns` when a retry window is known.

### Load test

```bash
python3 health/smoke_load_test.py
```

Ramps concurrency through Hermes-backed Tier 1/2 models (1→2→4→6→8→10) running 3 HumanEval problems each, while logging memory/swap/load-average/process-count every second to `/tmp/smoke_loadtest.log`. Run manually to check headroom before raising smoke-test concurrency on resource-constrained hosts; safe to Ctrl-C.

## Rate limit tracking

The router reads provider rate-limit headers on every successful response:

- **Groq**: `x-ratelimit-remaining-requests` + `x-ratelimit-reset-requests`
- **Cerebras**: `x-ratelimit-remaining-requests-minute`

When remaining hits 0, the model is held for the reset window duration. The next request automatically routes to the next candidate without burning a 429. State persists across restarts in `.rate_state.json`. Live state is visible at `GET /health`.

## Cron schedule

```
45 5 * * *    cron/daily.sh    # cheahjs sync + liveness probe
48 5 * * 1    cron/weekly.sh   # SWE-bench refresh + smoke test
```

Logs: `/tmp/llm-pool-daily.log`, `/tmp/llm-pool-weekly.log`
