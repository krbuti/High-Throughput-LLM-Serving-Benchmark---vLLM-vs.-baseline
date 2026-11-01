"""Async load generator for OpenAI-compatible completion servers.

Works against vLLM, the Hugging Face baseline in ``servers/hf`` and anything
else that speaks ``POST /v1/completions`` with SSE streaming.

Two load modes:

* closed loop (``--concurrency N``): N virtual users, each sends its next
  request as soon as the previous one finishes. Good for "how much can one GPU
  push at N parallel users".
* open loop (``--request-rate R``): Poisson arrivals at R req/s regardless of
  how fast the server answers. Good for showing queueing blow-ups.

Example::

    python -m bench.loadgen --base-url http://localhost:8000 \
        --model Qwen/Qwen2.5-7B-Instruct --concurrency 1 4 16 64 \
        --num-prompts 200 --input-len 512 --output-len 128 \
        --out results/vllm_default.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

import httpx

from bench.dataset import build_prompts
from bench.metrics import RequestResult, result_to_dict, summarize


async def send_one(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    ignore_eos: bool,
    t0: float,
) -> RequestResult:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        # vLLM extension (also honoured by our HF server): always generate
        # exactly max_tokens so both servers do the same amount of work.
        payload["ignore_eos"] = True

    start = time.perf_counter()
    res = RequestResult(ok=False, start=start - t0)
    first: float | None = None
    last = start
    chunks = 0
    usage = None
    try:
        async with client.stream("POST", url, json=payload) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode(errors="replace")
                res.error = f"HTTP {resp.status_code}: {body[:200]}"
                return res
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                msg = json.loads(data)
                if msg.get("usage"):
                    usage = msg["usage"]
                choices = msg.get("choices") or []
                if not choices or not choices[0].get("text"):
                    continue
                now = time.perf_counter()
                if first is None:
                    first = now
                else:
                    res.itl.append(now - last)
                last = now
                chunks += 1
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        res.error = f"{type(e).__name__}: {e}"
        return res

    end = time.perf_counter()
    if first is None:
        res.error = "no tokens received"
        return res
    res.ok = True
    res.ttft = first - start
    res.latency = end - start
    if usage:
        res.prompt_tokens = int(usage.get("prompt_tokens") or 0)
        res.output_tokens = int(usage.get("completion_tokens") or chunks)
    else:
        res.output_tokens = chunks  # one token per chunk is a good approximation
    return res


async def run_closed_loop(client, url, model, prompts, concurrency, max_tokens, ignore_eos):
    queue: asyncio.Queue[str] = asyncio.Queue()
    for p in prompts:
        queue.put_nowait(p)
    results: list[RequestResult] = []
    t0 = time.perf_counter()

    async def user():
        while True:
            try:
                p = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            results.append(await send_one(client, url, model, p, max_tokens, ignore_eos, t0))

    await asyncio.gather(*(user() for _ in range(concurrency)))
    return results, time.perf_counter() - t0


async def run_open_loop(client, url, model, prompts, rate, max_tokens, ignore_eos, seed):
    rng = random.Random(seed)
    t0 = time.perf_counter()
    tasks = []
    for p in prompts:
        tasks.append(asyncio.create_task(send_one(client, url, model, p, max_tokens, ignore_eos, t0)))
        await asyncio.sleep(rng.expovariate(rate))
    results = await asyncio.gather(*tasks)
    return list(results), time.perf_counter() - t0


async def warmup(client, url, model, prompt, n=2):
    for _ in range(n):
        await send_one(client, url, model, prompt, 8, False, time.perf_counter())


async def benchmark(args) -> dict:
    prompts = build_prompts(
        num_prompts=args.num_prompts,
        input_len=args.input_len,
        workload=args.workload,
        prefix_len=args.prefix_len,
        seed=args.seed,
        tokenizer=args.tokenizer,
    )
    texts = [p.text for p in prompts]
    url = args.base_url.rstrip("/") + "/v1/completions"
    max_conn = max(args.concurrency or [1]) + 16 if args.request_rate is None else 4096
    limits = httpx.Limits(max_connections=max_conn, max_keepalive_connections=max_conn)
    timeout = httpx.Timeout(args.timeout, connect=30.0)

    runs = []
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        if not args.no_warmup:
            # Use a prompt that does not share the benchmark prefix, so warmup
            # never pre-fills the prefix cache for the measured run.
            await warmup(client, url, args.model, "Hello, how are you?")

        levels = [("rate", r) for r in args.request_rate] if args.request_rate else [
            ("concurrency", c) for c in args.concurrency
        ]
        for kind, level in levels:
            if kind == "concurrency":
                res, wall = await run_closed_loop(
                    client, url, args.model, texts, level, args.output_len, not args.allow_eos
                )
            else:
                res, wall = await run_open_loop(
                    client, url, args.model, texts, level, args.output_len, not args.allow_eos, args.seed
                )
            summary = summarize(res, wall)
            summary[kind] = level
            runs.append({"summary": summary, "requests": [result_to_dict(r) for r in res] if args.save_requests else None})
            s = summary
            print(
                f"[{args.label}] {kind}={level:<5} ok={s['num_ok']}/{s['num_requests']} "
                f"out_tok/s={s['output_throughput_tok_s']:8.1f} "
                f"req/s={s['request_throughput_rps']:6.2f} "
                f"TTFT p50={s['ttft_s']['p50']*1e3:7.1f}ms p95={s['ttft_s']['p95']*1e3:7.1f}ms "
                f"e2e p95={s['e2e_latency_s']['p95']:6.2f}s "
                f"TPOT p50={s['tpot_s']['p50']*1e3:6.1f}ms",
                flush=True,
            )
            if summary["num_errors"]:
                print(f"    errors: {summary['errors']}", file=sys.stderr)

    return {
        "label": args.label,
        "base_url": args.base_url,
        "model": args.model,
        "workload": args.workload,
        "input_len": args.input_len,
        "prefix_len": args.prefix_len,
        "output_len": args.output_len,
        "num_prompts": args.num_prompts,
        "extra": json.loads(args.extra) if args.extra else {},
        "runs": runs,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--label", default="run")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16, 64])
    g.add_argument("--request-rate", type=float, nargs="+", default=None)
    p.add_argument("--num-prompts", type=int, default=200)
    p.add_argument("--input-len", type=int, default=512)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--workload", choices=["random", "shared-prefix"], default="random")
    p.add_argument("--prefix-len", type=int, default=0)
    p.add_argument("--tokenizer", default=None, help="HF tokenizer for exact prompt lengths")
    p.add_argument("--allow-eos", action="store_true", help="let the model stop early (default: ignore EOS)")
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--save-requests", action="store_true", help="store every request in the output JSON")
    p.add_argument("--extra", default=None, help="JSON blob of server settings to store with the results")
    p.add_argument("--out", type=Path, default=None)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = asyncio.run(benchmark(args))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"wrote {args.out}")
    return result


if __name__ == "__main__":
    main()
