"""Baseline: an OpenAI-compatible server on plain Hugging Face Transformers.

This is the "standard" way people serve a model before reaching for vLLM:
``model.generate()`` behind FastAPI. It deliberately has none of vLLM's tricks:

* KV cache is the default contiguous HF ``DynamicCache``: each sequence's
  cache is a dense tensor padded to the longest sequence in the batch, so
  memory is reserved for padding and nothing is shared between requests.
* Batching is *static*: a batch is formed from whatever is queued
  (up to ``HF_MAX_BATCH_SIZE``, waiting at most ``HF_BATCH_WAIT_MS``), then
  runs to completion. New requests wait for the whole batch to finish, and
  short requests sit idle in the batch until the longest one is done.
* No prefix caching.

``HF_MAX_BATCH_SIZE=1`` (the default) gives fully sequential serving, which is
what a naive ``pipeline()`` endpoint does.

Endpoints: ``/v1/models``, ``/v1/completions``, ``/v1/chat/completions``
(streaming and non-streaming), ``/health``, ``/metrics``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer

log = logging.getLogger("hf-server")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
SERVED_NAME = os.environ.get("SERVED_MODEL_NAME", MODEL)
DTYPE = os.environ.get("DTYPE", "bfloat16")
MAX_BATCH_SIZE = int(os.environ.get("HF_MAX_BATCH_SIZE", "1"))
BATCH_WAIT_MS = float(os.environ.get("HF_BATCH_WAIT_MS", "10"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "4096"))
ATTN_IMPL = os.environ.get("ATTN_IMPL", "sdpa")

# ---------------------------------------------------------------- metrics ---
_LAT_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 80, 160)
M_REQ = Counter("hf_requests_total", "Finished requests", ["status"])
M_PROMPT = Counter("hf_prompt_tokens_total", "Prompt tokens processed")
M_GEN = Counter("hf_generation_tokens_total", "Generated tokens")
M_RUNNING = Gauge("hf_num_requests_running", "Requests in the current batch")
M_WAITING = Gauge("hf_num_requests_waiting", "Requests queued behind the current batch")
M_TTFT = Histogram("hf_time_to_first_token_seconds", "Time to first token", buckets=_LAT_BUCKETS)
M_E2E = Histogram("hf_e2e_request_latency_seconds", "End-to-end request latency", buckets=_LAT_BUCKETS)
M_QUEUE = Histogram("hf_request_queue_time_seconds", "Time spent waiting for a batch slot", buckets=_LAT_BUCKETS)
M_BATCH = Histogram("hf_batch_size", "Sequences per generate() call", buckets=(1, 2, 4, 8, 16, 32, 64, 128))
M_GPU_MEM = Gauge("hf_gpu_memory_allocated_bytes", "torch.cuda.memory_allocated()")
M_GPU_RES = Gauge("hf_gpu_memory_reserved_bytes", "torch.cuda.memory_reserved()")


# ------------------------------------------------------------------ model ---
def _load():
    log.info("loading %s (dtype=%s, attn=%s)", MODEL, DTYPE, ATTN_IMPL)
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"  # decoder-only batching must pad on the left
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=getattr(torch, DTYPE) if device == "cuda" else torch.float32,
        attn_implementation=ATTN_IMPL,
    ).to(device)
    model.eval()
    return tok, model, device


tokenizer, model, DEVICE = _load()
EOS_IDS: set[int] = set()
for _eos in (model.generation_config.eos_token_id, tokenizer.eos_token_id):
    if isinstance(_eos, int):
        EOS_IDS.add(_eos)
    elif _eos:
        EOS_IDS.update(_eos)


# -------------------------------------------------------------- requests ---
@dataclass
class GenRequest:
    prompt_ids: list[int]
    max_tokens: int
    temperature: float
    top_p: float
    ignore_eos: bool
    stop: list[str]
    loop: asyncio.AbstractEventLoop
    out: asyncio.Queue  # receives str deltas, then a final dict
    arrived: float = field(default_factory=time.perf_counter)
    # Filled in by the worker thread:
    ids: list[int] = field(default_factory=list)
    text: str = ""
    emitted: int = 0  # chars of ``text`` already sent to the client
    prefix_offset: int = 0
    read_offset: int = 0
    done: bool = False
    cancelled: bool = False
    finish_reason: str = "length"
    first_token_at: float | None = None

    def emit(self, item: Any) -> None:
        self.loop.call_soon_threadsafe(self.out.put_nowait, item)

    def flush(self, final: bool = False) -> None:
        """Send unsent text, holding back a tail that could start a stop string."""
        end = len(self.text)
        if not final:
            for s in self.stop:
                for k in range(min(len(s) - 1, end), 0, -1):
                    if self.text.endswith(s[:k]):
                        end = min(end, len(self.text) - k)
                        break
        if end > self.emitted:
            self.emit(self.text[self.emitted : end])
            self.emitted = end


class BatchStreamer(BaseStreamer):
    """Receives one token per sequence per decode step from ``generate()``."""

    def __init__(self, batch: list[GenRequest]):
        self.batch = batch
        self.seen_prompt = False

    def put(self, value: torch.Tensor) -> None:
        if not self.seen_prompt:  # generate() first passes the prompt ids
            self.seen_prompt = True
            return
        toks = value.view(-1).tolist()
        now = time.perf_counter()
        for req, tok_id in zip(self.batch, toks):
            if req.done:
                continue
            if req.cancelled:
                req.done = True
                continue
            if tok_id in EOS_IDS and not req.ignore_eos:
                req.done, req.finish_reason = True, "stop"
                continue
            req.ids.append(tok_id)
            if req.first_token_at is None:
                req.first_token_at = now
                M_TTFT.observe(now - req.arrived)
            delta = self._decode_delta(req)
            if delta:
                req.text += delta
                hits = [req.text.find(s) for s in req.stop if s in req.text]
                if hits:  # the stop string may straddle several deltas
                    req.text = req.text[: min(hits)]
                    req.done, req.finish_reason = True, "stop"
                req.flush(final=req.done)
            if len(req.ids) >= req.max_tokens:
                req.done = True

    @staticmethod
    def _decode_delta(req: GenRequest) -> str:
        # Incremental detokenisation (same scheme as TGI/vLLM): only decode a
        # small window and hold back incomplete UTF-8 sequences.
        prefix = tokenizer.decode(req.ids[req.prefix_offset : req.read_offset], skip_special_tokens=True)
        full = tokenizer.decode(req.ids[req.prefix_offset :], skip_special_tokens=True)
        if len(full) > len(prefix) and not full.endswith("�"):
            req.prefix_offset, req.read_offset = req.read_offset, len(req.ids)
            return full[len(prefix) :]
        return ""

    def end(self) -> None:
        pass


class AllDone(StoppingCriteria):
    """Stop ``generate()`` as soon as every sequence in the batch is finished."""

    def __init__(self, batch: list[GenRequest]):
        self.batch = batch

    def __call__(self, input_ids, scores, **kwargs):
        done = all(r.done for r in self.batch)
        return torch.full((input_ids.shape[0],), done, dtype=torch.bool, device=input_ids.device)


_queue: "queue.Queue[GenRequest]" = queue.Queue()


def _next_batch() -> list[GenRequest]:
    batch = [_queue.get()]  # block until there is work
    deadline = time.perf_counter() + BATCH_WAIT_MS / 1000
    while len(batch) < MAX_BATCH_SIZE:
        remaining = deadline - time.perf_counter()
        try:
            batch.append(_queue.get(timeout=max(remaining, 0)) if remaining > 0 else _queue.get_nowait())
        except queue.Empty:
            break
    return [r for r in batch if not r.cancelled]


@torch.inference_mode()
def _run_batch(batch: list[GenRequest]) -> None:
    now = time.perf_counter()
    for r in batch:
        M_QUEUE.observe(now - r.arrived)
    M_RUNNING.set(len(batch))
    M_WAITING.set(_queue.qsize())
    M_BATCH.observe(len(batch))

    enc = tokenizer.pad({"input_ids": [r.prompt_ids for r in batch]}, return_tensors="pt").to(DEVICE)
    max_new = max(r.max_tokens for r in batch)
    min_new = max((r.max_tokens for r in batch if r.ignore_eos), default=0)
    # Sampling settings are per batch in HF; we take the first request's.
    sample = batch[0].temperature > 0
    gen_kwargs = dict(
        max_new_tokens=max_new,
        min_new_tokens=min_new or None,
        do_sample=sample,
        pad_token_id=tokenizer.pad_token_id,
        streamer=BatchStreamer(batch),
        stopping_criteria=StoppingCriteriaList([AllDone(batch)]),
    )
    if sample:
        gen_kwargs.update(temperature=batch[0].temperature, top_p=batch[0].top_p)
    else:
        gen_kwargs.update(temperature=None, top_p=None, top_k=None)

    try:
        model.generate(**enc, **gen_kwargs)
        error = None
    except Exception as e:  # e.g. CUDA OOM at large batch sizes
        log.exception("generate failed")
        error = f"{type(e).__name__}: {e}"
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    end = time.perf_counter()
    for r in batch:
        M_PROMPT.inc(len(r.prompt_ids))
        M_GEN.inc(len(r.ids))
        M_E2E.observe(end - r.arrived)
        M_REQ.labels(status="error" if error else "ok").inc()
        r.flush(final=True)
        r.emit({"error": error, "text": r.text, "finish_reason": r.finish_reason, "completion_tokens": len(r.ids),
                "prompt_tokens": len(r.prompt_ids)})
    M_RUNNING.set(0)
    if DEVICE == "cuda":
        M_GPU_MEM.set(torch.cuda.memory_allocated())
        M_GPU_RES.set(torch.cuda.memory_reserved())


def _worker() -> None:
    while True:
        batch = _next_batch()
        if batch:
            _run_batch(batch)


threading.Thread(target=_worker, daemon=True, name="generate-worker").start()

# -------------------------------------------------------------------- api ---
app = FastAPI(title="HF Transformers baseline")


def _submit(prompt_ids: list[int], body: dict) -> GenRequest:
    max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 16)
    if len(prompt_ids) + max_tokens > MAX_MODEL_LEN:
        raise HTTPException(400, f"prompt ({len(prompt_ids)}) + max_tokens ({max_tokens}) > {MAX_MODEL_LEN}")
    stop = body.get("stop") or []
    req = GenRequest(
        prompt_ids=prompt_ids,
        max_tokens=max_tokens,
        temperature=float(body.get("temperature", 1.0) or 0.0),
        top_p=float(body.get("top_p", 1.0) or 1.0),
        ignore_eos=bool(body.get("ignore_eos", False)),
        stop=[stop] if isinstance(stop, str) else list(stop),
        loop=asyncio.get_running_loop(),
        out=asyncio.Queue(),
    )
    _queue.put(req)
    M_WAITING.set(_queue.qsize())
    return req


async def _collect(req: GenRequest) -> tuple[str, dict]:
    while True:
        item = await req.out.get()
        if isinstance(item, dict):
            if item["error"]:
                raise HTTPException(500, item["error"])
            return item["text"], item


def _usage(final: dict) -> dict:
    return {
        "prompt_tokens": final["prompt_tokens"],
        "completion_tokens": final["completion_tokens"],
        "total_tokens": final["prompt_tokens"] + final["completion_tokens"],
    }


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


async def _stream(request: Request, req: GenRequest, make_chunk, include_usage: bool, rid: str, obj: str):
    try:
        first = True
        while True:
            item = await req.out.get()
            if isinstance(item, dict):
                if item["error"]:
                    yield _sse({"error": {"message": item["error"]}})
                    break
                yield _sse(make_chunk("", item["finish_reason"], False))
                if include_usage:
                    yield _sse({"id": rid, "object": obj, "created": int(time.time()), "model": SERVED_NAME,
                                "choices": [], "usage": _usage(item)})
                break
            yield _sse(make_chunk(item, None, first))
            first = False
        yield "data: [DONE]\n\n"
    finally:
        if not req.done:
            req.cancelled = True  # client went away; free the batch slot early


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    M_WAITING.set(_queue.qsize())
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": SERVED_NAME, "object": "model", "owned_by": "hf-baseline",
                                        "max_model_len": MAX_MODEL_LEN}]}


@app.post("/v1/completions")
async def completions(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    if isinstance(prompt, list):
        if len(prompt) != 1:
            raise HTTPException(400, "only a single prompt per request is supported")
        prompt = prompt[0]
    ids = tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
    req = _submit(ids, body)
    rid, created = f"cmpl-{uuid.uuid4().hex}", int(time.time())

    if body.get("stream"):
        def chunk(text, finish, _first):
            return {"id": rid, "object": "text_completion", "created": created, "model": SERVED_NAME,
                    "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish}]}

        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        return StreamingResponse(_stream(request, req, chunk, include_usage, rid, "text_completion"),
                                 media_type="text/event-stream")

    text, final = await _collect(req)
    return JSONResponse({"id": rid, "object": "text_completion", "created": created, "model": SERVED_NAME,
                         "choices": [{"index": 0, "text": text, "logprobs": None,
                                      "finish_reason": final["finish_reason"]}],
                         "usage": _usage(final)})


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    if body.get("tools"):
        raise HTTPException(400, "tool calling is not supported by the HF baseline; point the agent at vLLM")
    ids = tokenizer.apply_chat_template(body["messages"], add_generation_prompt=True, tokenize=True)
    if not isinstance(ids, list):  # some transformers versions return a BatchEncoding
        ids = list(ids["input_ids"])
    req = _submit(ids, body)
    rid, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())

    if body.get("stream"):
        def chunk(text, finish, first):
            delta = {"role": "assistant", "content": text} if first else ({"content": text} if text else {})
            return {"id": rid, "object": "chat.completion.chunk", "created": created, "model": SERVED_NAME,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        return StreamingResponse(_stream(request, req, chunk, include_usage, rid, "chat.completion.chunk"),
                                 media_type="text/event-stream")

    text, final = await _collect(req)
    return JSONResponse({"id": rid, "object": "chat.completion", "created": created, "model": SERVED_NAME,
                         "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                      "finish_reason": final["finish_reason"]}],
                         "usage": _usage(final)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8001")))
