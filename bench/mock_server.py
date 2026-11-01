"""A fake OpenAI-compatible streaming server for testing the harness without a GPU.

It models a server with ``--slots`` parallel decode slots, a prefill cost per
prompt token and a fixed time per decode step. It's a stand-in so the load
generator, sweep and plots can be tested end to end. It is not a model of vLLM.

    python -m bench.mock_server --port 8000 --slots 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse


def make_app(slots: int = 8, prefill_s_per_tok: float = 2e-5, step_s: float = 0.002) -> FastAPI:
    app = FastAPI()
    sem = asyncio.Semaphore(slots)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": "mock", "object": "model"}]}

    @app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()
        n_prompt = len(str(body.get("prompt", "")).split())
        n_out = int(body.get("max_tokens", 16))

        async def gen():
            async with sem:
                await asyncio.sleep(n_prompt * prefill_s_per_tok)
                for i in range(n_out):
                    chunk = {"id": "x", "object": "text_completion", "created": int(time.time()),
                             "model": "mock", "choices": [{"index": 0, "text": f" t{i}", "finish_reason": None}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
                    await asyncio.sleep(step_s)
            if (body.get("stream_options") or {}).get("include_usage"):
                usage = {"prompt_tokens": n_prompt, "completion_tokens": n_out, "total_tokens": n_prompt + n_out}
                yield f"data: {json.dumps({'id': 'x', 'choices': [], 'usage': usage})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


if __name__ == "__main__":
    import uvicorn

    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--slots", type=int, default=8)
    a = p.parse_args()
    uvicorn.run(make_app(a.slots), host="127.0.0.1", port=a.port, log_level="warning")
