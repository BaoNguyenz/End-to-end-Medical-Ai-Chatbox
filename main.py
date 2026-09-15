"""
GaleMed AI — Production-Ready RAG Evaluation Platform
=====================================================
Unified Master CLI Entry Point for GaleMed AI RAG Evaluation & Observability.

Combines:
    - Task 1: RAGAS Automated Evaluation (Faithfulness, Relevancy, Precision, Recall, Safety)
    - Task 2: LLM Observability & Cost Tracking (Langfuse Tracing, Token Pricing, PII Masking)
    - Task 3: Architecture Comparison & Ablation Study (Naive vs Full GaleMed)
    - Task 4: Integrated Platform & Automated Comprehensive Reporting

Usage:
    # Run full end-to-end evaluation & generate report:
    python main.py --run-experiments --generate-report --limit 10

    # Smoke test (mock mode):
    python main.py --mock --limit 5

    # Run only experiments:
    python main.py --run-experiments --limit 105

    # Run only report generation from saved results:
    python main.py --generate-report --results-dir output/experiments
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure RAGAS compatibility shim is active and API key exported
import src.evaluation  # pyrefly: ignore [unused-import]
from src.config import settings

if settings.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GaleMed AI: Production-Ready RAG Evaluation & Monitoring Platform"
    )
    parser.add_argument(
        "--run-experiments",
        action="store_true",
        help="Execute comparative experiments across RAG architectures.",
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate comprehensive Markdown comparison report and visual charts.",
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="Data/benchmarks/medical_benchmark_with_ground_truth.json",
        help="Path to evaluation benchmark JSON file.",
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
        help="Path for generated final Markdown report.",
    )
    parser.add_argument(
        "--charts-dir",
        type=str,
        default="output",
        help="Directory to save PNG comparison plots.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode (instant evaluation without external API calls).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Path to directory with existing results to generate report from.",
    )
    return parser.parse_args()


def main() -> None:
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    args = parse_args()

    # If neither flag is specified, default to running both
    run_exp = args.run_experiments
    gen_rep = args.generate_report
    if not run_exp and not gen_rep:
        run_exp = True
        gen_rep = True

    print("\n" + "=" * 76)
    print("  GALEMED AI - PRODUCTION-READY RAG EVALUATION & OBSERVABILITY PLATFORM")
    print("=" * 76)
    print(f"  Dataset    : {args.dataset}")
    print(f"  Limit      : {args.limit or 'All questions'}")
    print(f"  Mode       : {'MOCK (Fast Smoke Test)' if args.mock else 'REAL (Production Pipelines)'}")
    print(f"  Output     : {args.output}")
    print(f"  Charts Dir : {args.charts_dir}")
    print("=" * 76 + "\n")

    if args.results_dir and gen_rep and not run_exp:
        from compare import compare_from_results_dir
        compare_from_results_dir(args.results_dir, args.output, args.charts_dir)
        return

    from src.experiments.runner import ExperimentRunner
    from src.experiments.analysis import generate_charts, generate_markdown_report

    runner = ExperimentRunner(output_dir=args.charts_dir)

    if args.mock:
        from src.experiments.architectures import MockNaivePipeline, MockFullPipeline
        naive_pipeline = MockNaivePipeline()
        full_pipeline = MockFullPipeline()
    else:
        from src.experiments.architectures import NaiveRAGPipeline, FullGaleMedRAGPipeline
        naive_pipeline = NaiveRAGPipeline()
        full_pipeline = FullGaleMedRAGPipeline()

    from src.evaluation.dataset import load_benchmark

    if run_exp:
        logger.info("Loading benchmark from: %s", args.dataset)
        dataset = load_benchmark(args.dataset, verbose=False)
        questions = dataset.questions

        logger.info("Phase 1: Running comparative experiments...")
        naive_report = runner.run_architecture("naive_rag", naive_pipeline, questions=questions, limit=args.limit)
        full_report = runner.run_architecture("full_galemed", full_pipeline, questions=questions, limit=args.limit)
        comparison = runner.compare(naive_report, full_report)
        comparison.print_summary()

        # Always save JSON results to output/experiments/
        exp_dir = Path("output/experiments")
        exp_dir.mkdir(parents=True, exist_ok=True)
        comparison.save(exp_dir / "comparison_latest.json")



    if gen_rep:
        logger.info("Phase 2: Generating visual charts & comprehensive report...")
        chart_paths = generate_charts(comparison, output_dir=args.charts_dir)
        report_md = generate_markdown_report(comparison, chart_paths=chart_paths, output_path=args.output)
        logger.info("Report successfully generated: %s", args.output)

    print("\n" + "=" * 76)
    print("  [OK] PLATFORM EXECUTION FINISHED SUCCESSFULLY")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()

