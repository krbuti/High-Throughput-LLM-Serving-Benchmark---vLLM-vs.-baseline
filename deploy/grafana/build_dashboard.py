"""Generate deploy/grafana/dashboards/llm-serving.json.

Edit the panel list here and re-run rather than hand-editing the JSON:

    python deploy/grafana/build_dashboard.py

Queries use ``or`` across metric names that changed between vLLM releases
(``gpu_cache_usage_perc`` -> ``kv_cache_usage_perc``,
``time_per_output_token_seconds`` -> ``inter_token_latency_seconds``).
"""

import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "prometheus"}
# Reference categorical palette, fixed order.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

_id = 0


def _next_id():
    global _id
    _id += 1
    return _id


def q(expr, legend):
    return {"datasource": DS, "expr": expr, "legendFormat": legend, "refId": "A"}  # re-lettered per panel


def p_quant(metric, quant, legend, window="1m"):
    return q(f"histogram_quantile({quant}, sum by (le) (rate({metric}[{window}])))", legend)


def timeseries(title, targets, x, y, w=8, h=8, unit="short", desc=""):
    overrides = [
        {"matcher": {"id": "byName", "options": t["legendFormat"]},
         "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": C[i % len(C)]}}]}
        for i, t in enumerate(targets)
    ]
    for i, t in enumerate(targets):
        t["refId"] = chr(65 + i)
    return {
        "id": _next_id(), "type": "timeseries", "title": title, "description": desc, "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h}, "targets": targets,
        "fieldConfig": {"defaults": {"unit": unit, "custom": {"lineWidth": 2, "fillOpacity": 0,
                                                              "showPoints": "never"}},
                        "overrides": overrides},
        "options": {"legend": {"displayMode": "list", "placement": "bottom"},
                    "tooltip": {"mode": "multi", "sort": "desc"}},
    }


def stat(title, expr, x, y, unit="short", decimals=0, w=4):
    return {
        "id": _next_id(), "type": "stat", "title": title, "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": 4},
        "targets": [{"datasource": DS, "expr": expr, "refId": "A"}],
        "fieldConfig": {"defaults": {"unit": unit, "decimals": decimals,
                                     "color": {"mode": "fixed", "fixedColor": C[0]}}},
        "options": {"colorMode": "none", "graphMode": "area", "reduceOptions": {"calcs": ["lastNotNull"]}},
    }


def row(title, y):
    return {"id": _next_id(), "type": "row", "title": title, "collapsed": False,
            "gridPos": {"x": 0, "y": y, "w": 24, "h": 1}, "panels": []}


KV = "(vllm:kv_cache_usage_perc or vllm:gpu_cache_usage_perc)"
ITL = "(vllm:inter_token_latency_seconds_bucket or vllm:time_per_output_token_seconds_bucket)"

panels = [
    row("vLLM", 0),
    stat("Generation tokens/s", "sum(rate(vllm:generation_tokens_total[1m]))", 0, 1),
    stat("Requests running", "sum(vllm:num_requests_running)", 4, 1),
    stat("Requests waiting", "sum(vllm:num_requests_waiting)", 8, 1),
    stat("KV cache used", f"max({KV})", 12, 1, unit="percentunit", decimals=1),
    stat("p95 TTFT", "histogram_quantile(0.95, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m])))",
         16, 1, unit="s", decimals=2),
    stat("Prefix cache hit rate",
         "sum(rate(vllm:prefix_cache_hits_total[5m])) / clamp_min(sum(rate(vllm:prefix_cache_queries_total[5m])), 1)",
         20, 1, unit="percentunit", decimals=1),
    timeseries("Token throughput", [
        q("sum(rate(vllm:generation_tokens_total[1m]))", "generation tok/s"),
        q("sum(rate(vllm:prompt_tokens_total[1m]))", "prompt tok/s"),
    ], 0, 5, unit="short", desc="Tokens per second across all requests"),
    timeseries("Scheduler: running vs waiting", [
        q("sum(vllm:num_requests_running)", "running"),
        q("sum(vllm:num_requests_waiting)", "waiting"),
    ], 8, 5, desc="Continuous batching: running = sequences in the current batch"),
    timeseries("Paged KV cache usage", [
        q(f"max({KV})", "KV cache used"),
    ], 16, 5, unit="percentunit", desc="Fraction of PagedAttention blocks in use"),
    timeseries("Time to first token", [
        p_quant("vllm:time_to_first_token_seconds_bucket", 0.5, "p50"),
        p_quant("vllm:time_to_first_token_seconds_bucket", 0.95, "p95"),
        p_quant("vllm:time_to_first_token_seconds_bucket", 0.99, "p99"),
    ], 0, 13, unit="s"),
    timeseries("End-to-end request latency", [
        p_quant("vllm:e2e_request_latency_seconds_bucket", 0.5, "p50"),
        p_quant("vllm:e2e_request_latency_seconds_bucket", 0.95, "p95"),
        p_quant("vllm:e2e_request_latency_seconds_bucket", 0.99, "p99"),
    ], 8, 13, unit="s"),
    timeseries("Inter-token latency", [
        p_quant(ITL, 0.5, "p50"),
        p_quant(ITL, 0.95, "p95"),
    ], 16, 13, unit="s"),
    timeseries("Prefix cache hit rate", [
        q("sum(rate(vllm:prefix_cache_hits_total[1m])) / clamp_min(sum(rate(vllm:prefix_cache_queries_total[1m])), 1)",
          "hit rate"),
    ], 0, 21, unit="percentunit", desc="Share of prompt tokens served from cached KV blocks"),
    timeseries("Preemptions & finished requests", [
        q("sum(rate(vllm:num_preemptions_total[1m]))", "preemptions/s"),
        q("sum(rate(vllm:request_success_total[1m]))", "finished req/s"),
    ], 8, 21, desc="Preemptions climb when the KV cache is too small for the batch"),
    timeseries("Request queue time", [
        p_quant("vllm:request_queue_time_seconds_bucket", 0.5, "p50"),
        p_quant("vllm:request_queue_time_seconds_bucket", 0.95, "p95"),
    ], 16, 21, unit="s"),
    row("HF Transformers baseline", 29),
    timeseries("Token throughput", [
        q("sum(rate(hf_generation_tokens_total[1m]))", "generation tok/s"),
        q("sum(rate(hf_prompt_tokens_total[1m]))", "prompt tok/s"),
    ], 0, 30),
    timeseries("Running vs waiting", [
        q("sum(hf_num_requests_running)", "running"),
        q("sum(hf_num_requests_waiting)", "waiting"),
    ], 8, 30, desc="Static batching: waiting requests sit out the whole current batch"),
    timeseries("Batch size per generate()", [
        p_quant("hf_batch_size_bucket", 0.5, "p50"),
        p_quant("hf_batch_size_bucket", 0.95, "p95"),
    ], 16, 30),
    timeseries("Time to first token", [
        p_quant("hf_time_to_first_token_seconds_bucket", 0.5, "p50"),
        p_quant("hf_time_to_first_token_seconds_bucket", 0.95, "p95"),
    ], 0, 38, unit="s"),
    timeseries("End-to-end request latency", [
        p_quant("hf_e2e_request_latency_seconds_bucket", 0.5, "p50"),
        p_quant("hf_e2e_request_latency_seconds_bucket", 0.95, "p95"),
    ], 8, 38, unit="s"),
    timeseries("GPU memory (torch)", [
        q("max(hf_gpu_memory_allocated_bytes)", "allocated"),
        q("max(hf_gpu_memory_reserved_bytes)", "reserved"),
    ], 16, 38, unit="bytes"),
]

dashboard = {
    "uid": "llm-serving",
    "title": "LLM Serving: vLLM vs HF baseline",
    "tags": ["llm", "vllm"],
    "timezone": "browser",
    "schemaVersion": 39,
    "refresh": "5s",
    "time": {"from": "now-30m", "to": "now"},
    "panels": panels,
}

out = Path(__file__).parent / "dashboards" / "llm-serving.json"
out.write_text(json.dumps(dashboard, indent=2) + "\n")
print(f"wrote {out} ({len(panels)} panels)")
