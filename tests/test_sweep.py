from pathlib import Path

import subprocess

import pytest

from bench import sweep
from bench.sweep import deep_merge, docker_cmd, load_experiments, parse_vllm_kv, vllm_cli_args

ROOT = Path(__file__).resolve().parent.parent


def test_vllm_cli_args_booleans():
    assert vllm_cli_args({"enable-prefix-caching": False, "max_num_seqs": 8, "enforce-eager": True}) == [
        "--no-enable-prefix-caching", "--max-num-seqs", "8", "--enforce-eager"]


def test_deep_merge_overrides_nested_only():
    base = {"vllm_args": {"a": 1, "b": 2}, "bench": {"x": 1}}
    out = deep_merge(base, {"vllm_args": {"b": 3}})
    assert out == {"vllm_args": {"a": 1, "b": 3}, "bench": {"x": 1}}
    assert base["vllm_args"]["b"] == 2  # no mutation


def test_parse_vllm_kv_log_lines():
    logs = ("INFO 10-20 [kv_cache_utils.py:1087] GPU KV cache size: 254,672 tokens\n"
            "INFO 10-20 [kv_cache_utils.py:1091] Maximum concurrency for 4,096 tokens per request: 62.18x\n")
    assert parse_vllm_kv(logs) == {"kv_cache_tokens": 254672, "max_concurrency_tokens_per_req": 4096,
                                   "max_concurrency": 62.18}


def test_sweep_config_loads_and_builds_commands():
    exps = {e["name"]: e for e in load_experiments(ROOT / "configs" / "sweep.yaml")}
    assert "vllm_default" in exps and "hf_sequential" in exps
    off = exps["vllm_prefix_off"]
    assert off["vllm_args"]["gpu-memory-utilization"] == 0.90  # inherited
    assert off["bench"]["workload"] == "shared-prefix"
    cmd = docker_cmd(off)
    assert "--no-enable-prefix-caching" in cmd and "--model" in cmd
    hf = docker_cmd(exps["hf_static_bs8"])
    assert "HF_MAX_BATCH_SIZE=8" in hf and "8000:8001" in hf


def test_preflight_reports_missing_docker_and_gpu(monkeypatch):
    def missing(cmd, **kw):
        raise FileNotFoundError(cmd[0])
    monkeypatch.setattr(subprocess, "run", missing)
    problems = sweep.preflight()
    assert any("Docker is not installed" in p for p in problems)
    assert any("no NVIDIA GPU" in p for p in problems)


def test_sweep_exits_cleanly_without_gpu(monkeypatch, tmp_path):
    monkeypatch.setattr(sweep, "preflight", lambda: ["nvidia-smi not found"])
    with pytest.raises(SystemExit) as e:
        sweep.main([str(ROOT / "configs" / "sweep.yaml"), "--out-dir", str(tmp_path)])
    assert e.value.code == 2
