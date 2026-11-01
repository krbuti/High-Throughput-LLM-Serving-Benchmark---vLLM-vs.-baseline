from bench import loadgen, report


def test_report_end_to_end(mock_url, tmp_path):
    common = ["--base-url", mock_url, "--model", "mock", "--concurrency", "1", "4", "--num-prompts", "8",
              "--input-len", "32", "--output-len", "8", "--no-warmup"]
    for label, server in [("vllm_default", "vllm"), ("hf_sequential", "hf"), ("vllm_seqs_8", "vllm")]:
        loadgen.main(common + ["--label", label, "--extra", f'{{"server": "{server}"}}',
                               "--out", str(tmp_path / f"{label}.json")])
    out = tmp_path / "report"
    report.main(["--results", str(tmp_path), "--out", str(out)])
    assert (out / "summary.csv").exists()
    assert (out / "baseline_vs_vllm.png").exists()
    assert (out / "continuous_batching.png").exists()
    assert (out / "pareto.png").exists()
    md = (out / "REPORT.md").read_text()
    assert "Peak throughput" in md and "vllm_default" in md


def test_pareto_keeps_only_non_dominated():
    import pandas as pd
    df = pd.DataFrame({"e2e_p95_s": [1, 2, 3, 4], "output_tok_s": [100, 90, 300, 250], "config": list("abcd")})
    assert list(report.pareto(df).config) == ["a", "c"]
