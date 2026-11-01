"""A LangGraph tool-calling agent running on the self-hosted vLLM endpoint.

The model is reached through vLLM's OpenAI-compatible API, so the agent code is
the same as for a hosted model, only ``base_url`` changes. vLLM must be started
with ``--enable-auto-tool-choice --tool-call-parser hermes`` (Qwen2.5 format),
which docker-compose.yml does.

Tools:
  * ``calculator``             - safe arithmetic
  * ``benchmark_results``      - query this repo's benchmark results
  * ``live_server_metrics``    - read vLLM's Prometheus /metrics right now

    OPENAI_BASE_URL=http://localhost:8000/v1 python agent/agent.py
    python agent/agent.py -q "Which vLLM config had the best throughput under 10 s p95 latency?"
"""

from __future__ import annotations

import argparse
import ast
import json
import operator
import os
import re
from pathlib import Path
from typing import Annotated, TypedDict

import httpx
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-3B-Instruct")
RESULTS = Path(os.environ.get("RESULTS_DIR", Path(__file__).resolve().parent.parent / "results"))

# ------------------------------------------------------------------- tools ---
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        if isinstance(node.op, ast.Pow) and abs(_eval(node.right)) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("only numbers and + - * / // % ** are allowed")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. '2450 / 310' or '(1.2 + 3) * 4'."""
    try:
        return str(_eval(ast.parse(expression, mode="eval")))
    except Exception as e:  # report back to the model instead of crashing the graph
        return f"error: {e}"


def _load_summaries() -> list[dict]:
    rows = []
    for f in sorted(RESULTS.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        for run in d.get("runs", []):
            s = run["summary"]
            rows.append({
                "config": d["label"], "workload": d["workload"], "concurrency": s.get("concurrency"),
                "output_tok_s": round(s["output_throughput_tok_s"], 1),
                "ttft_p95_ms": round(s["ttft_s"]["p95"] * 1e3, 1),
                "e2e_p95_s": round(s["e2e_latency_s"]["p95"], 3),
                "errors": s["num_errors"],
                "kv_cache_tokens": d.get("extra", {}).get("kv_cache_tokens"),
            })
    return rows


@tool
def benchmark_results(config_filter: str = "", max_p95_latency_s: float = 0.0, top_k: int = 10) -> str:
    """Look up benchmark results, sorted by output throughput (highest first).

    Args:
        config_filter: substring of the config name, e.g. 'vllm', 'hf', 'mem', 'prefix'. Empty = all.
        max_p95_latency_s: if > 0, keep only runs whose p95 end-to-end latency is at most this.
        top_k: number of rows to return.
    """
    rows = [r for r in _load_summaries() if config_filter in r["config"]]
    if max_p95_latency_s > 0:
        rows = [r for r in rows if r["e2e_p95_s"] <= max_p95_latency_s]
    if not rows:
        return f"no results matched (results dir: {RESULTS})"
    rows.sort(key=lambda r: r["output_tok_s"], reverse=True)
    return json.dumps(rows[:top_k], indent=1)


_WANTED = ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc",
           "vllm:gpu_cache_usage_perc", "vllm:generation_tokens_total", "vllm:prompt_tokens_total",
           "vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total", "vllm:num_preemptions_total")


@tool
def live_server_metrics() -> str:
    """Read the vLLM server's current Prometheus metrics (queue depth, KV cache use, token counters)."""
    url = BASE_URL.rsplit("/v1", 1)[0] + "/metrics"
    try:
        text = httpx.get(url, timeout=5).text
    except httpx.HTTPError as e:
        return f"error reading {url}: {e}"
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z_:][\w:]*)(?:\{[^}]*\})?\s+([-\d.eE+naNinf]+)$", line)
        if m and m.group(1) in _WANTED:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    return json.dumps(out, indent=1) if out else "no vLLM metrics found"


TOOLS = [calculator, benchmark_results, live_server_metrics]

# ------------------------------------------------------------------- graph ---
SYSTEM = (
    "You are an LLM-serving performance assistant. Use the tools to answer from real data: "
    "benchmark_results for past benchmark runs, live_server_metrics for the running server, "
    "calculator for any arithmetic (speedups, ratios). Be concise and cite the numbers you used."
)


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def build_graph(llm: ChatOpenAI | None = None):
    llm = llm or ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                            temperature=0, max_tokens=1024)
    llm_tools = llm.bind_tools(TOOLS)

    def call_model(state: State):
        return {"messages": [llm_tools.invoke([SystemMessage(SYSTEM), *state["messages"]])]}

    g = StateGraph(State)
    g.add_node("agent", call_model)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


def run(graph, question: str, history: list | None = None, verbose: bool = True) -> list:
    messages = (history or []) + [HumanMessage(question)]
    final = None
    for step in graph.stream({"messages": messages}, {"recursion_limit": 12}, stream_mode="values"):
        final = step
        last = step["messages"][-1]
        if verbose and getattr(last, "tool_calls", None):
            for tc in last.tool_calls:
                print(f"  -> {tc['name']}({json.dumps(tc['args'])})")
    print(final["messages"][-1].content)
    return final["messages"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-q", "--question", help="ask one question and exit")
    a = p.parse_args()
    graph = build_graph()
    if a.question:
        run(graph, a.question)
        return
    print(f"agent on {MODEL} via {BASE_URL}  (ctrl-d to quit)")
    history: list = []
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            break
        if q:
            history = run(graph, q, history)


if __name__ == "__main__":
    main()
