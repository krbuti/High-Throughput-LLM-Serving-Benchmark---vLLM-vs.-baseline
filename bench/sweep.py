"""Run a grid of server configurations end to end.

For every experiment in the YAML config this:

1. starts the server container (vLLM or the HF baseline) with the given flags,
2. waits for ``/health``,
3. scrapes vLLM's startup log for the KV-cache size it allocated,
4. runs ``bench.loadgen`` at each concurrency level,
5. writes ``results/<name>.json`` and removes the container.

    python -m bench.sweep configs/sweep.yaml                  # everything
    python -m bench.sweep configs/sweep.yaml --only 'vllm_mem_*'
    python -m bench.sweep configs/sweep.yaml --dry-run        # print docker commands

Existing result files are skipped unless ``--force`` is given, so a sweep that
dies halfway (spot instance, OOM) resumes where it stopped.
"""

from __future__ import annotations

import argparse
import copy
import fnmatch
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import httpx
import yaml

from bench import loadgen

CONTAINER = "llm-bench-server"
HOST_PORT = 8000


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def vllm_cli_args(args: dict) -> list[str]:
    """``{"gpu-memory-utilization": 0.9, "enable-prefix-caching": False}`` -> CLI flags."""
    out = []
    for k, v in args.items():
        flag = "--" + k.replace("_", "-")
        if v is True:
            out.append(flag)
        elif v is False:
            out.append("--no-" + k.replace("_", "-"))
        elif v is None:
            continue
        else:
            out += [flag, str(v)]
    return out


def docker_cmd(exp: dict) -> list[str]:
    cache = os.path.expanduser(exp.get("hf_cache", "~/.cache/huggingface"))
    cmd = ["docker", "run", "-d", "--name", CONTAINER, "--gpus", "all", "--ipc=host",
           "-v", f"{cache}:/root/.cache/huggingface"]
    for var in ("HF_TOKEN",):
        if os.environ.get(var):
            cmd += ["-e", var]
    if exp["server"] == "vllm":
        cmd += ["-p", f"{HOST_PORT}:8000", exp["vllm_image"], "--model", exp["model"],
                "--served-model-name", exp["model"]]
        cmd += vllm_cli_args(exp.get("vllm_args", {}))
    elif exp["server"] == "hf":
        env = {"MODEL": exp["model"], **{k: str(v) for k, v in exp.get("hf_env", {}).items()}}
        for k, v in env.items():
            cmd += ["-e", f"{k}={v}"]
        cmd += ["-p", f"{HOST_PORT}:8001", exp["hf_image"]]
    else:
        raise ValueError(f"unknown server {exp['server']!r}")
    return cmd


def stop_container() -> None:
    subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)


def container_logs() -> str:
    p = subprocess.run(["docker", "logs", CONTAINER], capture_output=True, text=True)
    return p.stdout + p.stderr


def wait_healthy(base_url: str, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER],
                               capture_output=True, text=True).stdout.strip()
        if state == "false":
            raise RuntimeError("server container exited during startup:\n" + container_logs()[-4000:])
        try:
            if httpx.get(base_url + "/health", timeout=5).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(5)
    raise TimeoutError(f"server not healthy after {timeout}s:\n" + container_logs()[-4000:])


_KV_TOKENS = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")
_MAX_CONC = re.compile(r"Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([\d.]+)x")
_BLOCKS = re.compile(r"# (?:GPU|CUDA) blocks:\s*(\d+)")


def parse_vllm_kv(logs: str) -> dict:
    """Pull the KV-cache sizing that vLLM prints at startup."""
    info = {}
    if m := _KV_TOKENS.search(logs):
        info["kv_cache_tokens"] = int(m.group(1).replace(",", ""))
    if m := _MAX_CONC.search(logs):
        info["max_concurrency_tokens_per_req"] = int(m.group(1).replace(",", ""))
        info["max_concurrency"] = float(m.group(2))
    if m := _BLOCKS.search(logs):
        info["gpu_blocks"] = int(m.group(1))
    return info


def bench_argv(exp: dict, base_url: str, out: Path) -> list[str]:
    b = exp["bench"]
    argv = ["--base-url", base_url, "--model", exp["model"], "--label", exp["name"],
            "--num-prompts", str(b["num_prompts"]), "--input-len", str(b["input_len"]),
            "--output-len", str(b["output_len"]), "--workload", b.get("workload", "random"),
            "--prefix-len", str(b.get("prefix_len", 0)), "--out", str(out)]
    if b.get("request_rate"):
        argv += ["--request-rate", *map(str, b["request_rate"])]
    else:
        argv += ["--concurrency", *map(str, b["concurrency"])]
    if b.get("tokenizer"):
        argv += ["--tokenizer", b["tokenizer"]]
    if b.get("save_requests"):
        argv.append("--save-requests")
    return argv


def run_experiment(exp: dict, out_dir: Path, dry_run: bool, launch: bool, base_url: str) -> None:
    out = out_dir / f"{exp['name']}.json"
    cmd = docker_cmd(exp)
    print(f"\n=== {exp['name']} ===")
    print("  " + shlex.join(cmd))
    if dry_run:
        return

    kv = {}
    if launch:
        stop_container()
        subprocess.run(cmd, check=True, capture_output=True)
        try:
            wait_healthy(base_url, exp.get("startup_timeout", 1200))
        except Exception:
            stop_container()
            raise
        if exp["server"] == "vllm":
            kv = parse_vllm_kv(container_logs())
            print(f"  KV cache: {kv}")
    try:
        extra = {"server": exp["server"], "vllm_args": exp.get("vllm_args"), "hf_env": exp.get("hf_env"),
                 "description": exp.get("description", ""), **kv}
        loadgen.main(bench_argv(exp, base_url, out) + ["--extra", json.dumps(extra)])
    finally:
        if launch:
            (out_dir / "logs").mkdir(exist_ok=True)
            (out_dir / "logs" / f"{exp['name']}.log").write_text(container_logs())
            stop_container()


def load_experiments(path: Path) -> list[dict]:
    cfg = yaml.safe_load(path.read_text())
    defaults = cfg.get("defaults", {})
    exps = []
    for e in cfg["experiments"]:
        merged = deep_merge(defaults, e)
        merged.setdefault("server", "vllm")
        exps.append(merged)
    return exps


def preflight() -> list[str]:
    """Return human-readable problems that would make every experiment fail."""
    problems = []
    try:
        if subprocess.run(["docker", "info"], capture_output=True, timeout=30).returncode != 0:
            problems.append("Docker is installed but the daemon isn't reachable (is it running? is your user in "
                            "the 'docker' group?)")
    except FileNotFoundError:
        problems.append("Docker is not installed")
    except subprocess.TimeoutExpired:
        problems.append("'docker info' timed out")
    try:
        if subprocess.run(["nvidia-smi"], capture_output=True, timeout=30).returncode != 0:
            problems.append("nvidia-smi failed: the NVIDIA driver isn't working")
    except FileNotFoundError:
        problems.append("nvidia-smi not found: this machine has no NVIDIA GPU/driver")
    except subprocess.TimeoutExpired:
        problems.append("'nvidia-smi' timed out")
    return problems


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=Path)
    p.add_argument("--only", nargs="*", default=None, help="glob(s) on experiment name")
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    p.add_argument("--force", action="store_true", help="re-run experiments that already have results")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-launch", action="store_true",
                   help="don't start containers; benchmark whatever is already on --base-url")
    p.add_argument("--base-url", default=f"http://localhost:{HOST_PORT}")
    a = p.parse_args(argv)

    if not (a.dry_run or a.no_launch) and (problems := preflight()):
        print("Can't run the sweep on this machine:", file=sys.stderr)
        for pr in problems:
            print(f"  - {pr}", file=sys.stderr)
        print("It needs a Linux GPU host with the NVIDIA driver, Docker and the NVIDIA Container Toolkit. "
              "See 'Quick start' in README.md. (--dry-run prints the commands without running them.)",
              file=sys.stderr)
        sys.exit(2)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for exp in load_experiments(a.config):
        if a.only and not any(fnmatch.fnmatch(exp["name"], g) for g in a.only):
            continue
        if (a.out_dir / f"{exp['name']}.json").exists() and not (a.force or a.dry_run):
            print(f"skip {exp['name']} (results exist)")
            continue
        try:
            run_experiment(exp, a.out_dir, a.dry_run, not a.no_launch, a.base_url)
        except Exception as e:  # keep the sweep going
            print(f"!!! {exp['name']} failed: {e}", file=sys.stderr)
            failed.append(exp["name"])
    if failed:
        print(f"\nfailed: {failed}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
