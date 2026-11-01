import json

from bench import loadgen


def test_closed_loop_against_mock(mock_url, tmp_path):
    out = tmp_path / "r.json"
    res = loadgen.main(["--base-url", mock_url, "--model", "mock", "--label", "t", "--concurrency", "1", "8",
                        "--num-prompts", "16", "--input-len", "64", "--output-len", "20", "--save-requests",
                        "--out", str(out)])
    assert json.loads(out.read_text())["label"] == "t"
    s1, s8 = (r["summary"] for r in res["runs"])
    for s in (s1, s8):
        assert s["num_ok"] == 16 and s["num_errors"] == 0
        assert s["total_output_tokens"] == 16 * 20  # exact count from usage chunk
        assert s["ttft_s"]["p50"] > 0
        assert s["e2e_latency_s"]["p50"] >= s["ttft_s"]["p50"]
    # The mock has 4 parallel slots, so 8 users must beat 1 user on throughput.
    assert s8["output_throughput_tok_s"] > 2 * s1["output_throughput_tok_s"]
    req = res["runs"][0]["requests"][0]
    assert len(req["itl"]) == 19


def test_open_loop_against_mock(mock_url):
    res = loadgen.main(["--base-url", mock_url, "--model", "mock", "--request-rate", "50",
                        "--num-prompts", "10", "--input-len", "32", "--output-len", "5", "--no-warmup"])
    s = res["runs"][0]["summary"]
    assert s["rate"] == 50 and s["num_ok"] == 10


def test_connection_errors_are_recorded_not_raised():
    res = loadgen.main(["--base-url", "http://127.0.0.1:9", "--model", "x", "--concurrency", "2",
                        "--num-prompts", "3", "--input-len", "8", "--output-len", "2", "--no-warmup"])
    s = res["runs"][0]["summary"]
    assert s["num_errors"] == 3 and s["num_ok"] == 0
