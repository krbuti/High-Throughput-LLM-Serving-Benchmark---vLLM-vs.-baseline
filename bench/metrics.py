"""Per-request records and aggregate metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class RequestResult:
    ok: bool
    prompt_tokens: int = 0
    output_tokens: int = 0
    ttft: float | None = None  # seconds from send to first generated text
    latency: float | None = None  # seconds from send to last byte
    itl: list[float] = field(default_factory=list)  # gaps between streamed chunks
    start: float = 0.0  # perf_counter at send, relative to run start
    error: str | None = None

    @property
    def tpot(self) -> float | None:
        """Time per output token after the first one (decode speed seen by a user)."""
        if self.ttft is None or self.latency is None or self.output_tokens < 2:
            return None
        return (self.latency - self.ttft) / (self.output_tokens - 1)


def _pct(values: list[float], qs=(50, 90, 95, 99)) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), **{f"p{q}": float("nan") for q in qs}}
    arr = np.asarray(values, dtype=float)
    out = {"mean": float(arr.mean())}
    for q in qs:
        out[f"p{q}"] = float(np.percentile(arr, q))
    return out


def summarize(results: list[RequestResult], wall_time: float) -> dict:
    ok = [r for r in results if r.ok]
    out_tok = sum(r.output_tokens for r in ok)
    in_tok = sum(r.prompt_tokens for r in ok)
    all_itl = [g for r in ok for g in r.itl]
    errors: dict[str, int] = {}
    for r in results:
        if not r.ok:
            key = (r.error or "unknown")[:120]
            errors[key] = errors.get(key, 0) + 1

    return {
        "num_requests": len(results),
        "num_ok": len(ok),
        "num_errors": len(results) - len(ok),
        "errors": errors,
        "wall_time_s": wall_time,
        "request_throughput_rps": len(ok) / wall_time if wall_time > 0 else 0.0,
        "output_throughput_tok_s": out_tok / wall_time if wall_time > 0 else 0.0,
        "total_throughput_tok_s": (in_tok + out_tok) / wall_time if wall_time > 0 else 0.0,
        "total_prompt_tokens": in_tok,
        "total_output_tokens": out_tok,
        "ttft_s": _pct([r.ttft for r in ok if r.ttft is not None]),
        "tpot_s": _pct([t for r in ok if (t := r.tpot) is not None]),
        "itl_s": _pct(all_itl),
        "e2e_latency_s": _pct([r.latency for r in ok if r.latency is not None]),
    }


def result_to_dict(r: RequestResult) -> dict:
    d = asdict(r)
    d["tpot"] = r.tpot
    return d
