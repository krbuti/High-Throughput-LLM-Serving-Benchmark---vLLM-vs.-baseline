"""Run the real HF baseline server on CPU with a tiny random model.

Skipped unless torch and transformers are installed (they are not in
requirements-dev.txt because the CUDA wheels are large).
"""

import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from tests.conftest import _free_port  # noqa: E402
from tests.tiny_model import make_tiny_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def hf_url(tmp_path_factory):
    model = make_tiny_model(tmp_path_factory.mktemp("tiny"))
    port = _free_port()
    env = {**os.environ, "MODEL": str(model), "SERVED_MODEL_NAME": "tiny", "HF_MAX_BATCH_SIZE": "4",
           "HF_BATCH_WAIT_MS": "50", "PORT": str(port), "HOST": "127.0.0.1", "LOG_LEVEL": "WARNING"}
    proc = subprocess.Popen([sys.executable, str(ROOT / "servers/hf/server.py")], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    url = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            if httpx.get(url + "/health").status_code == 200:
                break
        except httpx.HTTPError:
            if proc.poll() is not None:
                raise RuntimeError(proc.stderr.read().decode())
            time.sleep(0.5)
    yield url
    proc.terminate()
    proc.wait(timeout=10)


def _complete(url, **body):
    return httpx.post(url + "/v1/completions", json={"model": "tiny", "temperature": 0, **body}, timeout=60).json()


def _stream(url, **body):
    text, usage = "", None
    with httpx.stream("POST", url + "/v1/completions", timeout=60,
                      json={"model": "tiny", "temperature": 0, "stream": True,
                            "stream_options": {"include_usage": True}, **body}) as r:
        for line in r.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                m = json.loads(line[6:])
                usage = m.get("usage") or usage
                if m["choices"]:
                    text += m["choices"][0]["text"]
    return text, usage


def test_ignore_eos_generates_exactly_max_tokens(hf_url):
    r = _complete(hf_url, prompt="the model", max_tokens=17, ignore_eos=True)
    assert r["usage"]["completion_tokens"] == 17


def test_batched_streaming_matches_sequential(hf_url):
    """Static batching with left padding must not change greedy outputs."""
    cases = [("the river city", 30), ("music painting theatre poetry", 7), ("data", 19), ("forest ocean", 12)]
    ref = [_complete(hf_url, prompt=p, max_tokens=n) for p, n in cases]
    with cf.ThreadPoolExecutor(4) as ex:
        got = list(ex.map(lambda c: _stream(hf_url, prompt=c[0], max_tokens=c[1]), cases))
    for r, (text, usage), (_, n) in zip(ref, got, cases):
        assert text == r["choices"][0]["text"]
        assert usage["completion_tokens"] == r["usage"]["completion_tokens"] <= n


def test_stop_string_spanning_deltas_is_never_sent(hf_url):
    for prompt in ["the river city", "data network storage", "music painting", "forest ocean"]:
        full = _complete(hf_url, prompt=prompt, max_tokens=40)["choices"][0]["text"]
        stop = re.search(r"[a-z]{3,}", full[8:]).group(0)
        want = full[: full.index(stop)]
        r = _complete(hf_url, prompt=prompt, max_tokens=40, stop=[stop])
        assert r["choices"][0]["finish_reason"] == "stop"
        assert r["choices"][0]["text"] == want
        assert _stream(hf_url, prompt=prompt, max_tokens=40, stop=[stop])[0] == want


def test_chat_and_errors(hf_url):
    r = httpx.post(hf_url + "/v1/chat/completions", json={
        "model": "tiny", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}).json()
    assert r["choices"][0]["message"]["role"] == "assistant"
    assert httpx.post(hf_url + "/v1/completions", json={"prompt": "x", "max_tokens": 10**6}).status_code == 400
    assert "hf_generation_tokens_total" in httpx.get(hf_url + "/metrics").text


def test_client_disconnect_frees_the_batch(hf_url):
    with httpx.stream("POST", hf_url + "/v1/completions", timeout=60, json={
            "model": "tiny", "prompt": "x", "max_tokens": 3500, "ignore_eos": True, "stream": True}) as r:
        for i, _ in enumerate(r.iter_lines()):
            if i > 6:
                break
    t = time.time()
    _complete(hf_url, prompt="y", max_tokens=3)
    assert time.time() - t < 5
