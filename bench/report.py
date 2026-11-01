"""Turn ``results/*.json`` into charts, a CSV and a Markdown report.

    python -m bench.report --results results --out results/report

Each chart is only drawn when the experiments it needs are present, so this
works on a partial sweep.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Validated reference palette (light mode). Categorical slots in fixed order;
# an ordinal blue ramp (steps 250 -> 700) for ordered parameter sweeps.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]

# Fixed identity colours so a config is the same colour in every chart.
FIXED = {
    "vllm_default": SLOTS[0],
    "hf_sequential": SLOTS[1],
    "hf_static_bs8": SLOTS[2],
    "hf_static_bs32": SLOTS[3],
    "vllm_prefix_on": SLOTS[0],
    "vllm_prefix_off": SLOTS[6],
    "hf_prefix_bs8": SLOTS[2],
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left",
    "font.size": 9.5, "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 6,
})


_COMMA = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}" if abs(v) >= 1000 else f"{v:g}")


def load(results_dir: Path) -> tuple[pd.DataFrame, dict]:
    rows, meta = [], {}
    for f in sorted(results_dir.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if "runs" not in d:
            continue
        meta[d["label"]] = d
        for run in d["runs"]:
            s = run["summary"]
            rows.append({
                "config": d["label"],
                "server": d.get("extra", {}).get("server", "vllm" if d["label"].startswith("vllm") else "hf"),
                "workload": d["workload"],
                "concurrency": s.get("concurrency"),
                "request_rate": s.get("rate"),
                "ok": s["num_ok"], "errors": s["num_errors"],
                "output_tok_s": s["output_throughput_tok_s"],
                "req_s": s["request_throughput_rps"],
                "ttft_p50_ms": s["ttft_s"]["p50"] * 1e3, "ttft_p95_ms": s["ttft_s"]["p95"] * 1e3,
                "tpot_p50_ms": s["tpot_s"]["p50"] * 1e3, "tpot_p95_ms": s["tpot_s"]["p95"] * 1e3,
                "e2e_p50_s": s["e2e_latency_s"]["p50"], "e2e_p95_s": s["e2e_latency_s"]["p95"],
                "e2e_p99_s": s["e2e_latency_s"]["p99"],
                "kv_cache_tokens": d.get("extra", {}).get("kv_cache_tokens"),
                "max_concurrency": d.get("extra", {}).get("max_concurrency"),
            })
    return pd.DataFrame(rows), meta


def _end_label(ax, x, y, text):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center",
                fontsize=8.5, color=INK_2)


def _line_panels(df, configs, colors, title, out, labels=None):
    """Throughput, p95 TTFT and p95 e2e latency vs concurrency, one panel each."""
    present = [c for c in configs if c in set(df.config)]
    if len(present) < 2:
        return None
    labels = labels or {}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    panels = [("output_tok_s", "Output throughput (tokens/s)", False),
              ("ttft_p95_ms", "p95 time to first token (ms, log)", True),
              ("e2e_p95_s", "p95 end-to-end latency (s, log)", True)]
    for ax, (col, ylabel, log) in zip(axes, panels):
        for c in present:
            d = df[df.config == c].sort_values("concurrency")
            ax.plot(d.concurrency, d[col], marker="o", color=colors[c], label=labels.get(c, c))
            if len(present) <= 4 and col == "output_tok_s":
                last = d.iloc[-1]
                _end_label(ax, last.concurrency, last[col], labels.get(c, c))
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Concurrent users")
        ax.set_title(ylabel)
        if log:
            ax.set_yscale("log")
        else:
            ax.yaxis.set_major_formatter(_COMMA)
        levels = sorted(df[df.config.isin(present)].concurrency.dropna().unique())
        ax.set_xticks(levels, [f"{int(v)}" for v in levels])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    lo, hi = axes[0].get_xlim()
    axes[0].set_xlim(lo, hi * 2.2)  # room for the end labels
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper center", ncol=min(len(lbls), 6), bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title, x=0.01, y=1.08, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def chart_baseline(df, out_dir):
    configs = ["vllm_default", "hf_static_bs32", "hf_static_bs8", "hf_sequential"]
    labels = {"vllm_default": "vLLM", "hf_static_bs32": "HF static bs=32",
              "hf_static_bs8": "HF static bs=8", "hf_sequential": "HF sequential"}
    return _line_panels(df, configs, FIXED, "vLLM vs Hugging Face Transformers baseline",
                        out_dir / "baseline_vs_vllm.png", labels)


def _ordered(df, pattern, key):
    found = sorted({c for c in df.config if re.fullmatch(pattern, c)}, key=key)
    return found


def chart_continuous_batching(df, out_dir):
    configs = _ordered(df, r"vllm_seqs_\d+", lambda c: int(c.split("_")[-1]))
    if "vllm_default" in set(df.config):
        configs.append("vllm_default")
    colors = {c: BLUE_RAMP[min(i, len(BLUE_RAMP) - 1)] for i, c in enumerate(configs)}
    labels = {c: f"max-num-seqs={c.split('_')[-1]}" for c in configs}
    labels["vllm_default"] = "max-num-seqs=256"
    return _line_panels(df, configs, colors, "Continuous batching: effect of max-num-seqs",
                        out_dir / "continuous_batching.png", labels)


def chart_memory(df, out_dir):
    configs = _ordered(df, r"vllm_mem_[\d.]+", lambda c: float(c.split("_")[-1]))
    if len(configs) < 2:
        return None
    rows = []
    for c in configs:
        d = df[df.config == c]
        top = d.loc[d.concurrency.idxmax()]
        rows.append((float(c.split("_")[-1]), top.kv_cache_tokens, top.output_tok_s, top.e2e_p95_s,
                     int(top.concurrency)))
    m = pd.DataFrame(rows, columns=["util", "kv", "tok_s", "e2e_p95", "conc"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    x = [f"{u:.2f}" for u in m.util]
    for ax, col, title in [(axes[0], "kv", "KV cache capacity (tokens)"),
                           (axes[1], "tok_s", f"Output tokens/s at {m.conc.iloc[0]} users"),
                           (axes[2], "e2e_p95", f"p95 e2e latency (s) at {m.conc.iloc[0]} users")]:
        vals = m[col].fillna(0)
        bars = ax.bar(x, vals, color=SLOTS[0], width=0.6, edgecolor=SURFACE, linewidth=2)
        ax.bar_label(bars, labels=[f"{v:,.0f}" if v >= 100 else f"{v:.2f}" for v in vals],
                     fontsize=8.5, color=INK_2, padding=2)
        ax.set_title(title)
        ax.yaxis.set_major_formatter(_COMMA)
        ax.set_xlabel("--gpu-memory-utilization")
        ax.grid(axis="x", visible=False)
    fig.suptitle("GPU memory utilisation -> paged KV cache size -> throughput", x=0.01, y=1.04,
                 ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = out_dir / "gpu_memory.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def chart_prefix(df, out_dir):
    configs = [c for c in ["vllm_prefix_on", "vllm_prefix_off", "hf_prefix_bs8"] if c in set(df.config)]
    if len(configs) < 2:
        return None
    labels = {"vllm_prefix_on": "vLLM\nprefix cache on", "vllm_prefix_off": "vLLM\nprefix cache off",
              "hf_prefix_bs8": "HF static\nbs=8"}
    # Compare at the highest concurrency every config ran.
    conc = min(df[df.config == c].concurrency.max() for c in configs)
    d = df[(df.config.isin(configs)) & (df.concurrency == conc)].set_index("config").loc[configs]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, col, title in [(axes[0], "ttft_p50_ms", "p50 TTFT (ms)"), (axes[1], "ttft_p95_ms", "p95 TTFT (ms)"),
                           (axes[2], "output_tok_s", "Output tokens/s")]:
        bars = ax.bar([labels[c] for c in configs], d[col], color=[FIXED[c] for c in configs],
                      width=0.6, edgecolor=SURFACE, linewidth=2)
        ax.bar_label(bars, labels=[f"{v:,.0f}" for v in d[col]], fontsize=8.5, color=INK_2, padding=2)
        ax.set_title(title)
        ax.yaxis.set_major_formatter(_COMMA)
        ax.grid(axis="x", visible=False)
    fig.suptitle(f"Prefix caching on a shared-prefix workload ({int(conc)} concurrent users)",
                 x=0.01, y=1.04, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = out_dir / "prefix_caching.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def pareto(df: pd.DataFrame) -> pd.DataFrame:
    """Points not dominated on (higher throughput, lower p95 latency)."""
    d = df.sort_values(["e2e_p95_s", "output_tok_s"], ascending=[True, False])
    front, best = [], -1.0
    for _, r in d.iterrows():
        if r.output_tok_s > best:
            front.append(r)
            best = r.output_tok_s
    return pd.DataFrame(front)


def chart_pareto(df, out_dir):
    d = df[(df.workload == "random") & (df.errors == 0)]
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    for server, color, name in [("hf", SLOTS[1], "HF Transformers"), ("vllm", SLOTS[0], "vLLM")]:
        s = d[d.server == server]
        ax.scatter(s.e2e_p95_s, s.output_tok_s, s=36, color=color, edgecolor=SURFACE, linewidth=1.5,
                   label=name, zorder=3)
    front = pareto(d)
    ax.plot(front.e2e_p95_s, front.output_tok_s, color=INK_2, linewidth=1, linestyle="--", zorder=2,
            label="Pareto frontier")
    ax.scatter(front.e2e_p95_s, front.output_tok_s, s=90, facecolor="none", edgecolor=INK, linewidth=1.2, zorder=4)
    # Labels go to the right of each frontier point, nudged apart vertically.
    last_y = None
    for _, r in front.sort_values("output_tok_s").iterrows():
        ypix = ax.transData.transform((r.e2e_p95_s, r.output_tok_s))[1]
        dy = 0 if last_y is None or ypix - last_y > 14 else 14 - (ypix - last_y)
        ax.annotate(f"{r.config} @{int(r.concurrency)} users", (r.e2e_p95_s, r.output_tok_s),
                    xytext=(9, dy - 3), textcoords="offset points", fontsize=7.5, color=INK_2)
        last_y = ypix + dy
    ax.set_xscale("log")
    ax.set_xlabel("p95 end-to-end latency (s, log)")
    ax.set_ylabel("Output tokens/s")
    ax.yaxis.set_major_formatter(_COMMA)
    ax.set_title("Throughput vs tail latency, every config and concurrency level")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out = out_dir / "pareto.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _fmt_table(d: pd.DataFrame, cols: list[str]) -> str:
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for _, r in d.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("" if pd.isna(v) else (f"{v:,.0f}" if abs(v) >= 100 else f"{v:.2f}"))
            else:
                cells.append(str(v))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def write_report(df, charts, meta, out_dir, latency_slo):
    lines = ["# Benchmark report", ""]
    if meta:
        any_meta = next(iter(meta.values()))
        lines += [f"Model: `{any_meta['model']}`", ""]

    rnd = df[df.workload == "random"]
    hf, vl = rnd[rnd.server == "hf"], rnd[rnd.server == "vllm"]
    if not hf.empty and not vl.empty:
        best_hf, best_vl = hf.loc[hf.output_tok_s.idxmax()], vl.loc[vl.output_tok_s.idxmax()]
        lines += ["## Headline", "",
                  f"- Peak throughput: **vLLM {best_vl.output_tok_s:,.0f} tok/s** "
                  f"(`{best_vl.config}`, {int(best_vl.concurrency)} users) vs **HF {best_hf.output_tok_s:,.0f} tok/s** "
                  f"(`{best_hf.config}`, {int(best_hf.concurrency)} users): "
                  f"**{best_vl.output_tok_s / best_hf.output_tok_s:.1f}x**."]
        common = sorted(set(hf.concurrency) & set(vl[vl.config == "vllm_default"].concurrency))
        if common:
            c = common[-1]
            h = hf[hf.concurrency == c].sort_values("output_tok_s").iloc[-1]
            v = vl[(vl.config == "vllm_default") & (vl.concurrency == c)].iloc[0]
            lines.append(f"- At {int(c)} users: p95 TTFT {v.ttft_p95_ms:,.0f} ms (vLLM) vs {h.ttft_p95_ms:,.0f} ms "
                         f"(`{h.config}`), p95 latency {v.e2e_p95_s:.2f}s vs {h.e2e_p95_s:.2f}s.")
        ok = vl[vl.e2e_p95_s <= latency_slo]
        if not ok.empty:
            b = ok.loc[ok.output_tok_s.idxmax()]
            lines.append(f"- Best vLLM throughput with p95 latency <= {latency_slo:g}s: **{b.output_tok_s:,.0f} tok/s** "
                         f"with `{b.config}` at {int(b.concurrency)} users.")
        lines.append("")

    for title, path in charts:
        lines += [f"## {title}", "", f"![{title}]({path.name})", ""]

    lines += ["## Pareto-optimal configurations (random workload)", "",
              _fmt_table(pareto(rnd[rnd.errors == 0]),
                         ["config", "concurrency", "output_tok_s", "ttft_p95_ms", "e2e_p95_s"]) if not rnd.empty else "",
              "## All results", "",
              _fmt_table(df.sort_values(["config", "concurrency"]),
                         ["config", "workload", "concurrency", "ok", "errors", "output_tok_s", "req_s",
                          "ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms", "e2e_p50_s", "e2e_p95_s", "kv_cache_tokens"])]
    path = out_dir / "REPORT.md"
    path.write_text("\n".join(lines))
    return path


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path, default=Path("results/report"))
    p.add_argument("--latency-slo", type=float, default=10.0, help="p95 e2e latency budget in seconds")
    a = p.parse_args(argv)
    df, meta = load(a.results)
    if df.empty:
        raise SystemExit(f"no results in {a.results}")
    a.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out / "summary.csv", index=False)
    charts = []
    for title, fn in [("vLLM vs baseline", chart_baseline), ("Continuous batching", chart_continuous_batching),
                      ("GPU memory utilisation", chart_memory), ("Prefix caching", chart_prefix),
                      ("Throughput vs latency", chart_pareto)]:
        if (path := fn(df, a.out)) is not None:
            charts.append((title, path))
    report = write_report(df, charts, meta, a.out, a.latency_slo)
    print(f"wrote {a.out / 'summary.csv'}, {report} and {len(charts)} charts")


if __name__ == "__main__":
    main()
