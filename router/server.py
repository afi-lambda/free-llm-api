#!/usr/bin/env python3
"""
OpenAI-compatible proxy server.
Point any tool at http://localhost:8000 with any API key string.

  LLM_BASE_URL=http://localhost:8000/v1  (for tools expecting /v1 prefix)
  OPENAI_BASE_URL=http://localhost:8000
  OPENAI_API_KEY=free-pool              (ignored — we inject real keys)
"""

import json
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from router.client import AllProvidersExhausted, complete
from router.pool import candidates, summary
from router.rate_tracker import RateTracker

_tracker = RateTracker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = candidates()
    alive = [m for m in pool if m.get("alive") is True]
    print(f"Free-LLM pool ready — {summary()}")
    if alive:
        print(f"  Top model: {alive[0]['id']} (T{alive[0].get('tier','?')})")
    yield


app = FastAPI(title="free-llm-pool", lifespan=lifespan)


# ── Request / Response models ────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    max_tokens: int = 2048
    temperature: float = 0.0
    stream: bool = False


# ── Routes ───────────────────────────────────────────────────────────────────

@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, request: Request):
    messages = [m.model_dump() for m in req.messages]

    try:
        if req.stream:
            stream = await complete(
                messages,
                model_hint=req.model,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                stream=True,
            )
            return StreamingResponse(stream, media_type="text/event-stream")
        else:
            result = await complete(
                messages,
                model_hint=req.model,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                stream=False,
            )
            return JSONResponse(result)

    except AllProvidersExhausted as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
@app.get("/models")
async def list_models():
    pool = candidates()
    return {
        "object": "list",
        "data": [
            {
                "id": m["id"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": m["provider"],
                "tier": m.get("tier"),
                "swebench_score": m.get("swebench_score"),
                "alive": m.get("alive"),
                "latency_ms": m.get("latency_ms"),
            }
            for m in pool
        ],
    }


@app.get("/health")
async def health():
    pool = candidates()
    alive = [m for m in pool if m.get("alive") is True]
    skipped = [m for m in pool if m.get("alive") is None]
    return {
        "status": "ok" if alive else "degraded",
        "pool_size": len(pool),
        "alive": len(alive),
        "skipped_no_key": len(skipped),
        "best": alive[0]["id"] if alive else None,
        "rate_tracker": _tracker.status(),
    }


@app.get("/")
async def root():
    return {"service": "free-llm-pool", "summary": summary()}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("FREE_LLM_PORT", 8000))
    uvicorn.run("router.server:app", host="127.0.0.1", port=port, reload=False)
