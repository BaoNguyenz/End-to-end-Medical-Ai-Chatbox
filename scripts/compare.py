"""
GaleMed AI — Architecture Comparison CLI
========================================
CLI tool to coordinate and compare multiple RAG architectures (Naive vs Full GaleMed).
Can run live comparisons or aggregate existing evaluated results.

Usage:
    # Live comparative evaluation:
    python scripts/compare.py --limit 10 --output output/report.md
    python scripts/compare.py --mock --limit 10 --output output/report.md

    # Aggregate existing evaluate.py results:
    python scripts/compare.py --results-dir results/ --output output/report.md
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure RAGAS compatibility shim is active and API key exported
import src.evaluation  # pyrefly: ignore [unused-import]
from src.config import settings

if settings.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("compare")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Naive RAG vs Full GaleMed RAG architectures."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Path to directory containing naive_results.json and full_results.json. If provided, skips live run.",
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="Data/benchmarks/medical_benchmark_with_ground_truth.json",
        help="Path to benchmark JSON dataset.",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Max number of benchmark questions to evaluate.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output/report.md",
        help="Path to output markdown report file.",
    )
    parser.add_argument(
        "--charts-dir",
        type=str,
        default="output",
        help="Directory to save generated chart PNGs.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock pipelines (instant run without external services).",
    )
    return parser.parse_args()


def compare_from_results_dir(results_dir: str, output_path: str, charts_dir: str) -> None:
    from src.experiments.runner import ComparisonResult
    from src.experiments.analysis import generate_charts, generate_markdown_report

    rdir = Path(results_dir)
    naive_file = rdir / "naive_results.json"
    full_file = rdir / "full_results.json"

    if not naive_file.exists() or not full_file.exists():
        raise FileNotFoundError(
            f"Could not find required files in {results_dir}: "
            f"ensure {naive_file.name} and {full_file.name} exist."
        )

    with naive_file.open("r", encoding="utf-8") as f:
        n_data = json.load(f)
    with full_file.open("r", encoding="utf-8") as f:
        f_data = json.load(f)

    naive_scores = {**n_data.get("metrics", {}), **n_data.get("performance", {})}
    full_scores = {**f_data.get("metrics", {}), **f_data.get("performance", {})}

    deltas = {}
    for k, v in full_scores.items():
        if k in naive_scores and isinstance(v, (int, float)) and isinstance(naive_scores[k], (int, float)):
            deltas[k] = round(v - naive_scores[k], 4)

    comp = ComparisonResult(
        naive_architecture=n_data.get("architecture", "Naive RAG"),
        full_architecture=f_data.get("architecture", "Full GaleMed RAG"),
        naive_scores=naive_scores,
        full_scores=full_scores,
        deltas=deltas,
        category_breakdown={},
        num_questions=len(n_data.get("queries", [])),
    )

    comp.print_summary()
    chart_paths = generate_charts(comp, output_dir=charts_dir)
    generate_markdown_report(comp, chart_paths=chart_paths, output_path=output_path)
    logger.info("Comparison complete from results dir. Report written to: %s", output_path)


def run_live_comparison(
    dataset_path: str,
    limit: Optional[int],
    output_path: str,
    charts_dir: str,
    mock: bool = False,
) -> None:
    from src.experiments.runner import ExperimentRunner
    from src.experiments.analysis import generate_charts, generate_markdown_report
    from src.evaluation.dataset import load_benchmark

    runner = ExperimentRunner(output_dir=charts_dir)

    if mock:
        from src.experiments.architectures import MockNaivePipeline, MockFullPipeline
        naive_pipeline = MockNaivePipeline()
        full_pipeline = MockFullPipeline()
    else:
        from src.experiments.architectures import NaiveRAGPipeline, FullGaleMedRAGPipeline
        naive_pipeline = NaiveRAGPipeline()
        full_pipeline = FullGaleMedRAGPipeline()

    logger.info("Loading benchmark from: %s", dataset_path)
    dataset = load_benchmark(dataset_path, verbose=False)
    questions = dataset.questions

    logger.info("Running comparative experiment (limit=%s, mock=%s)...", limit, mock)
    naive_report = runner.run_architecture("naive_rag", naive_pipeline, questions=questions, limit=limit)
    full_report = runner.run_architecture("full_galemed", full_pipeline, questions=questions, limit=limit)
    comparison = runner.compare(naive_report, full_report)

    comparison.print_summary()

    # Generate visual charts
    chart_paths = generate_charts(comparison, output_dir=charts_dir)

    # Generate Markdown report
    generate_markdown_report(comparison, chart_paths=chart_paths, output_path=output_path)
    logger.info("Comparison complete. Report generated at: %s", output_path)


def main() -> None:
    args = parse_args()
    if args.results_dir:
        compare_from_results_dir(
            results_dir=args.results_dir,
            output_path=args.output,
            charts_dir=args.charts_dir,
        )
    else:
        run_live_comparison(
            dataset_path=args.dataset,
            limit=args.limit,
            output_path=args.output,
            charts_dir=args.charts_dir,
            mock=args.mock,
        )


if __name__ == "__main__":
    main()
