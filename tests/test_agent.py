import json

import pytest

pytest.importorskip("langgraph")

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

import agent.agent as ag  # noqa: E402


class ScriptedLLM(BaseChatModel):
    """Calls a tool on the first turn, then answers using the tool output."""

    @property
    def _llm_type(self):
        return "scripted"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        if not tool_msgs:
            msg = AIMessage(content="", tool_calls=[{"name": "calculator", "args": {"expression": "2400/300"},
                                                     "id": "c1", "type": "tool_call"}])
        else:
            msg = AIMessage(content=f"speedup is {tool_msgs[-1].content}x")
        return ChatResult(generations=[ChatGeneration(message=msg)])


def test_graph_runs_tool_loop():
    graph = ag.build_graph(ScriptedLLM())
    out = graph.invoke({"messages": [("user", "how much faster?")]})
    assert out["messages"][-1].content == "speedup is 8.0x"


def test_calculator_rejects_code():
    assert ag.calculator.invoke({"expression": "__import__('os').system('ls')"}).startswith("error")
    assert ag.calculator.invoke({"expression": "(3 + 5) * 2 ** 3"}) == "64"


def test_benchmark_results_tool(tmp_path, monkeypatch):
    run = {"summary": {"concurrency": 16, "output_throughput_tok_s": 1234.5, "ttft_s": {"p95": 0.2},
                       "e2e_latency_s": {"p95": 3.0}, "num_errors": 0}}
    (tmp_path / "vllm_x.json").write_text(json.dumps({"label": "vllm_x", "workload": "random", "runs": [run]}))
    monkeypatch.setattr(ag, "RESULTS", tmp_path)
    rows = json.loads(ag.benchmark_results.invoke({"config_filter": "vllm", "max_p95_latency_s": 5}))
    assert rows[0]["output_tok_s"] == 1234.5
    assert "no results" in ag.benchmark_results.invoke({"config_filter": "hf"})
