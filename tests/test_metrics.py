import math

from bench.dataset import build_prompts
from bench.metrics import RequestResult, summarize


def test_tpot_excludes_first_token():
    r = RequestResult(ok=True, output_tokens=11, ttft=0.5, latency=1.5)
    assert math.isclose(r.tpot, 0.1)


def test_tpot_undefined_for_single_token():
    assert RequestResult(ok=True, output_tokens=1, ttft=0.1, latency=0.1).tpot is None


def test_summarize_throughput_and_percentiles():
    ok = [RequestResult(ok=True, prompt_tokens=10, output_tokens=100, ttft=t, latency=t + 1) for t in
          [0.1 * i for i in range(1, 11)]]
    bad = [RequestResult(ok=False, error="HTTP 500: boom")]
    s = summarize(ok + bad, wall_time=2.0)
    assert s["num_ok"] == 10 and s["num_errors"] == 1
    assert s["output_throughput_tok_s"] == 500.0
    assert s["total_throughput_tok_s"] == 550.0
    assert math.isclose(s["ttft_s"]["p50"], 0.55)
    assert s["ttft_s"]["p95"] > s["ttft_s"]["p50"]
    assert s["errors"] == {"HTTP 500: boom": 1}


def test_summarize_all_failed_is_nan_not_crash():
    s = summarize([RequestResult(ok=False, error="x")], 1.0)
    assert math.isnan(s["ttft_s"]["p95"])


def test_random_prompts_are_unique_and_sized():
    ps = build_prompts(20, input_len=200, seed=1)
    assert len({p.text for p in ps}) == 20
    assert all(150 <= p.approx_tokens <= 260 for p in ps)


def test_shared_prefix_prompts_share_exactly_the_prefix():
    ps = build_prompts(10, input_len=600, workload="shared-prefix", prefix_len=500)
    a, b = ps[0].text, ps[1].text
    common = 0
    while a[common] == b[common]:
        common += 1
    assert common > 0.7 * len(a)  # long shared prefix ...
    assert a != b  # ... but different suffixes


def test_prompts_are_deterministic():
    assert build_prompts(5, 100, seed=3)[0].text == build_prompts(5, 100, seed=3)[0].text
