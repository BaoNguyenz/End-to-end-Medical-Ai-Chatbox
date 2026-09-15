"""
GaleMed AI — Experiment Analysis & Visualization
=================================================
Generates statistical comparison charts between Naive RAG and Full GaleMed RAG.

Output charts (saved to output/experiments/):
    1. radar_chart.png  — Spider/radar chart: 6 metrics side-by-side
    2. bar_chart.png    — Grouped bar chart: per-category breakdown

Usage:
    from src.experiments.analysis import generate_charts, generate_markdown_report
    paths = generate_charts(comparison, output_dir="output/experiments")
    report_md = generate_markdown_report(comparison, chart_paths=paths)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Metric display labels (ordered for charts)
_METRIC_LABELS = {
    "faithfulness":       "Faithfulness",
    "answer_relevancy":   "Answer\nRelevancy",
    "context_precision":  "Context\nPrecision",
    "context_recall":     "Context\nRecall",
    "medical_safety":     "Medical\nSafety",
    "negative_rejection": "Negative\nRejection",
}

_NAIVE_COLOR = "#5B8DB8"     # Steel blue
_FULL_COLOR  = "#2ECC71"     # Emerald green
_DELTA_POS   = "#27AE60"     # Positive delta (green)
_DELTA_NEG   = "#E74C3C"     # Negative delta (red)


def _import_matplotlib():
    """Import matplotlib with a clear error message if missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend (works on servers)
        import matplotlib.pyplot as plt
        import numpy as np
        return plt, np
    except ImportError as exc:
        raise ImportError(
            "matplotlib and numpy are required for chart generation.\n"
            "Install with: pip install matplotlib numpy"
        ) from exc


def generate_radar_chart(
    comparison,
    output_path: str | Path,
    title: str = "RAG Architecture Comparison — Core Metrics",
) -> Path:
    """
    Generate a radar/spider chart comparing 6 metrics across 2 architectures.

    Args:
        comparison:   ComparisonResult from ExperimentRunner.compare().
        output_path:  Path to save the PNG.
        title:        Chart title.

    Returns:
        Absolute path to the saved PNG file.
    """
    plt, np = _import_matplotlib()

    metric_keys = list(_METRIC_LABELS.keys())
    labels = list(_METRIC_LABELS.values())
    N = len(metric_keys)

    naive_vals = [comparison.naive_scores.get(m) or 0.0 for m in metric_keys]
    full_vals  = [comparison.full_scores.get(m)  or 0.0 for m in metric_keys]

    # Angles for each metric (evenly spaced around circle)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]  # close the polygon

    naive_vals_plot = naive_vals + naive_vals[:1]
    full_vals_plot  = full_vals  + full_vals[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Style: dark background
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#16213E")

    # Draw grid lines
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="#AAAAAA", size=8)
    ax.yaxis.set_tick_params(labelsize=8)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color="white", size=9, fontweight="bold")

    # Grid styling
    ax.grid(color="#333366", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.spines["polar"].set_color("#333366")

    # Plot Naive RAG
    ax.plot(angles, naive_vals_plot, "o-", linewidth=2, color=_NAIVE_COLOR, label="Naive RAG", markersize=6)
    ax.fill(angles, naive_vals_plot, alpha=0.15, color=_NAIVE_COLOR)

    # Plot Full GaleMed
    ax.plot(angles, full_vals_plot, "s-", linewidth=2, color=_FULL_COLOR, label="Full GaleMed RAG", markersize=6)
    ax.fill(angles, full_vals_plot, alpha=0.20, color=_FULL_COLOR)

    # Value annotations on each vertex
    for angle, n_val, f_val in zip(angles[:-1], naive_vals, full_vals):
        ax.annotate(
            f"{n_val:.2f}", xy=(angle, n_val), fontsize=7.5,
            color=_NAIVE_COLOR, fontweight="bold",
            xytext=(0, 8), textcoords="offset points", ha="center",
        )
        ax.annotate(
            f"{f_val:.2f}", xy=(angle, f_val), fontsize=7.5,
            color=_FULL_COLOR, fontweight="bold",
            xytext=(0, -14), textcoords="offset points", ha="center",
        )

    # Legend
    legend = ax.legend(
        loc="upper right", bbox_to_anchor=(1.35, 1.15),
        fontsize=10, framealpha=0.3,
        labelcolor="white", facecolor="#1A1A2E", edgecolor="#333366",
    )

    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=20)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Radar chart saved: %s", out_path)
    return out_path.resolve()


def generate_bar_chart(
    comparison,
    output_path: str | Path,
    metric: str = "faithfulness",
    title: Optional[str] = None,
) -> Path:
    """
    Generate a grouped bar chart showing per-category breakdown for one metric.

    Args:
        comparison:   ComparisonResult from ExperimentRunner.compare().
        output_path:  Path to save the PNG.
        metric:       Which metric to visualise per category.
        title:        Chart title (auto-generated if None).

    Returns:
        Absolute path to the saved PNG file.
    """
    plt, np = _import_matplotlib()

    if not comparison.category_breakdown:
        logger.warning("No category breakdown data — skipping bar chart.")
        return Path(output_path)

    categories = sorted(comparison.category_breakdown.keys())
    naive_vals = [comparison.category_breakdown[c].get(metric, {}).get("naive") or 0.0 for c in categories]
    full_vals  = [comparison.category_breakdown[c].get(metric, {}).get("full")  or 0.0 for c in categories]

    x = np.arange(len(categories))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#16213E")

    # Draw bars
    bars_naive = ax.bar(x - bar_width / 2, naive_vals, bar_width, label="Naive RAG",       color=_NAIVE_COLOR, alpha=0.85, edgecolor="white", linewidth=0.5)
    bars_full  = ax.bar(x + bar_width / 2, full_vals,  bar_width, label="Full GaleMed RAG", color=_FULL_COLOR,  alpha=0.85, edgecolor="white", linewidth=0.5)

    # Value labels on each bar
    for bar in bars_naive:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}", ha="center", va="bottom", fontsize=8, color=_NAIVE_COLOR, fontweight="bold")

    for bar in bars_full:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}", ha="center", va="bottom", fontsize=8, color=_FULL_COLOR, fontweight="bold")

    # Delta annotations between bars
    for i, (n, f) in enumerate(zip(naive_vals, full_vals)):
        delta = f - n
        color = _DELTA_POS if delta >= 0 else _DELTA_NEG
        sign = "+" if delta >= 0 else ""
        ax.text(i, max(n, f) + 0.06, f"{sign}{delta:.2f}", ha="center", va="bottom", fontsize=8.5, color=color, fontweight="bold")

    # Styling
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in categories], color="white", fontsize=9)
    ax.set_ylabel(_METRIC_LABELS.get(metric, metric), color="white", fontsize=11)
    ax.set_xlabel("Medical Specialty / Category", color="white", fontsize=11)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333366")
    ax.yaxis.set_tick_params(colors="white")
    ax.grid(axis="y", color="#333366", linestyle="--", linewidth=0.7, alpha=0.6)

    title = title or f"Per-Category {_METRIC_LABELS.get(metric, metric).replace(chr(10), ' ')}: Naive RAG vs Full GaleMed"
    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=15)

    legend = ax.legend(fontsize=10, framealpha=0.3, labelcolor="white", facecolor="#1A1A2E", edgecolor="#333366")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Bar chart saved: %s", out_path)
    return out_path.resolve()


def generate_charts(
    comparison,
    output_dir: str | Path = "output/experiments",
) -> dict[str, Path]:
    """
    Generate all comparison charts (radar + faithfulness bar).

    Args:
        comparison:  ComparisonResult from ExperimentRunner.compare().
        output_dir:  Directory to save PNGs.

    Returns:
        Dict mapping chart name to absolute path: {"radar": Path, "bar_faithfulness": Path}
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    try:
        paths["radar"] = generate_radar_chart(
            comparison,
            output_path=out / "radar_chart.png",
        )
    except Exception as exc:
        logger.error("Radar chart failed: %s", exc)

    try:
        paths["bar_faithfulness"] = generate_bar_chart(
            comparison,
            output_path=out / "bar_faithfulness.png",
            metric="faithfulness",
        )
    except Exception as exc:
        logger.error("Bar chart failed: %s", exc)

    return paths


def generate_markdown_report(
    comparison,
    chart_paths: Optional[dict[str, Path]] = None,
    output_path: Optional[str | Path] = None,
) -> str:
    """
    Generate a Markdown report summarising the architecture comparison.

    Args:
        comparison:   ComparisonResult from ExperimentRunner.compare().
        chart_paths:  Dict of chart names → file paths (from generate_charts()).
        output_path:  If provided, write the Markdown to this file.

    Returns:
        Markdown string.
    """
    def _f(v) -> str:
        return f"{v:.4f}" if v is not None else "N/A"

    def _delta_str(v) -> str:
        if v is None:
            return "N/A"
        sign = "+" if v >= 0 else ""
        emoji = "✅" if v > 0.01 else ("⚠️" if v < -0.01 else "➡️")
        return f"{emoji} {sign}{v:.4f}"

    metrics = [
        ("Faithfulness",       "faithfulness"),
        ("Answer Relevancy",   "answer_relevancy"),
        ("Context Precision",  "context_precision"),
        ("Context Recall",     "context_recall"),
        ("Medical Safety",     "medical_safety"),
        ("Negative Rejection", "negative_rejection"),
        ("Avg Latency (ms)",   "avg_latency_ms"),
    ]

    lines = [
        "# GaleMed RAG Architecture Comparison Report",
        "",
        f"**Baseline:** `{comparison.naive_architecture}`  ",
        f"**Enhanced:** `{comparison.full_architecture}`",
        "",
        "---",
        "",
        "## Overall Metric Summary",
        "",
        f"| Metric | {comparison.naive_architecture} | {comparison.full_architecture} | Delta |",
        "|:---|:---:|:---:|:---:|",
    ]

    for label, key in metrics:
        n = comparison.naive_scores.get(key)
        f = comparison.full_scores.get(key)
        d = comparison.deltas.get(key)
        lines.append(f"| {label} | {_f(n)} | {_f(f)} | {_delta_str(d)} |")

    n_cost = comparison.naive_scores.get("total_cost_usd", 0.0)
    f_cost = comparison.full_scores.get("total_cost_usd", 0.0)
    lines += [
        f"| Total Cost (USD) | ${n_cost:.4f} | ${f_cost:.4f} | — |",
        "",
        "---",
        "",
    ]

    # Charts section
    if chart_paths:
        lines += ["## Visualization Charts", ""]
        for name, path in chart_paths.items():
            lines.append(f"### {name.replace('_', ' ').title()}")
            lines.append(f"![{name}]({path})")
            lines.append("")

    # Category breakdown table
    if comparison.category_breakdown:
        lines += [
            "## Per-Category Faithfulness Breakdown",
            "",
            "| Category | Naive RAG | Full GaleMed | Delta |",
            "|:---|:---:|:---:|:---:|",
        ]
        for cat, metrics_data in sorted(comparison.category_breakdown.items()):
            fdata = metrics_data.get("faithfulness", {})
            n = fdata.get("naive")
            f = fdata.get("full")
            d = fdata.get("delta")
            lines.append(f"| {cat} | {_f(n)} | {_f(f)} | {_delta_str(d)} |")
        lines.append("")

    lines += [
        "---",
        "",
        "*Generated by GaleMed AI Evaluation System — Task 3: Architecture Comparison*",
    ]

    md = "\n".join(lines)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        logger.info("Markdown report saved: %s", out)

    return md
