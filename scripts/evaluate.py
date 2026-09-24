"""
GaleMed AI — Single Architecture Evaluator CLI
==============================================
CLI entry point to evaluate a single RAG architecture on the benchmark dataset.

Usage:
    python scripts/evaluate.py --architecture naive --limit 5
    python scripts/evaluate.py --architecture full --limit 105 --output results/
    python scripts/evaluate.py --architecture naive --mock --limit 10
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

# Setup stdout encoding for Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a RAG architecture on the medical benchmark dataset."
    )
    parser.add_argument(
        "--architecture", "-a",
        type=str,
        default="full",
        choices=["naive", "full", "advanced"],
        help="Architecture to evaluate: 'naive' (Baseline) or 'full'/'advanced' (GaleMed production).",
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
        help="Max number of questions to evaluate (None for all).",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results",
        help="Output directory to save evaluation results and reports.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock pipeline (simulates retrieval/LLM without external API/Qdrant dependencies).",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default="eval_user",
        help="User ID for observability tracing.",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Session ID for observability tracing.",
    )
    return parser.parse_args()


class TracedPipelineWrapper:
    """Wraps a pipeline callable to add Langfuse observability tracing on each call."""

    def __init__(self, inner_pipeline, architecture_name: str, user_id: str, session_id: str):
        self.inner = inner_pipeline
        self.arch = architecture_name
        self.user_id = user_id
        self.session_id = session_id
        from src.observability.langfuse_tracker import get_tracer
        self.tracer = get_tracer()

    def __call__(self, *, query: str) -> dict:
        t0 = time.perf_counter()
        with self.tracer.trace(
            name=f"eval_{self.arch}",
            user_id=self.user_id,
            session_id=self.session_id,
            input={"query": query},
            metadata={"architecture": self.arch},
        ) as root_trace:
            res = self.inner(query=query)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            with root_trace.span(
                f"{self.arch}_query",
                input={"query": query},
                metadata={
                    "latency_ms": latency_ms,
                    "cost_usd": res.get("cost_usd", 0.0),
                },
            ) as s:
                s.set_output({"answer": res.get("answer", "")[:120]})

            return res


def run_evaluation(
    architecture: str,
    dataset_path: str,
    limit: Optional[int] = None,
    output_dir: str = "results",
    mock: bool = False,
    user_id: str = "eval_user",
    session_id: Optional[str] = None,
):
    from src.evaluation.dataset import load_benchmark
    from src.evaluation.ragas_evaluator import RagasEvaluator
    from src.experiments.architectures import (
        MockNaivePipeline,
        MockFullPipeline,
        NaiveRAGPipeline,
        FullGaleMedRAGPipeline,
    )

    # Normalize name
    arch_norm = "full" if architecture in ("full", "advanced") else "naive"
    session_id = session_id or f"eval_{arch_norm}_{int(time.time())}"

    # Load dataset
    data_path = Path(dataset_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    logger.info("Loading dataset from: %s (limit: %s)", dataset_path, limit)
    dataset = load_benchmark(data_path, verbose=False)
    logger.info("Loaded %d questions.", dataset.total)

    # Initialize raw pipeline
    if mock:
        raw_pipeline = MockNaivePipeline() if arch_norm == "naive" else MockFullPipeline()
        logger.info("Using MOCK pipeline for %s", arch_norm)
    else:
        raw_pipeline = NaiveRAGPipeline() if arch_norm == "naive" else FullGaleMedRAGPipeline()
        logger.info("Using REAL pipeline for %s", arch_norm)

    # Wrap with observability tracer
    pipeline = TracedPipelineWrapper(
        inner_pipeline=raw_pipeline,
        architecture_name=arch_norm,
        user_id=user_id,
        session_id=session_id,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / f"checkpoint_{arch_norm}.json"

    # Execute via RagasEvaluator
    evaluator = RagasEvaluator(
        pipeline=pipeline,
        architecture=arch_norm,
        checkpoint_path=checkpoint_path,
        batch_size=5,
    )

    logger.info("Starting RAGAS evaluation for architecture: %s ...", arch_norm)
    report = evaluator.run(
        questions=dataset.questions,
        limit=limit,
        save_results_path=out_dir / f"{arch_norm}_results.json",
    )

    report.print_summary()

    # Also save formatted markdown report
    def _fmt(val: Optional[float]) -> str:
        return f"{val:.4f}" if val is not None else "N/A"

    lat_str = f"- **Average Latency:** {report.avg_latency_ms:.1f} ms" if report.avg_latency_ms else "- **Average Latency:** N/A"

    md_file = out_dir / f"{arch_norm}_report.md"
    md_lines = [
        f"# Evaluation Report: {arch_norm.upper()} RAG",
        "",
        f"- **Dataset:** `{dataset_path}` ({report.evaluated_questions}/{report.total_questions} questions)",
        f"- **Session ID:** `{session_id}`",
        f"- **Mode:** {'MOCK' if mock else 'REAL'}",
        "",
        "## RAGAS Core Metrics",
        "",
        "| Metric | Score |",
        "|:---|:---:|",
        f"| Faithfulness | {_fmt(report.avg_faithfulness)} |",
        f"| Answer Relevancy | {_fmt(report.avg_answer_relevancy)} |",
        f"| Context Precision | {_fmt(report.avg_context_precision)} |",
        f"| Context Recall | {_fmt(report.avg_context_recall)} |",
        "",
        "## Medical Safety Metrics",
        "",
        "| Metric | Score |",
        "|:---|:---:|",
        f"| Medical Safety | {_fmt(report.avg_medical_safety)} |",
        f"| Negative Rejection | {_fmt(report.avg_negative_rejection)} |",
        "",
        "## Performance & Cost",
        "",
        lat_str,
        f"- **Total Estimated Cost:** ${report.total_cost_usd:.4f}",
        "",
        "---",
        f"*Generated by evaluate.py at {time.strftime('%Y-%m-%d %H:%M:%S')}*",
    ]
    md_file.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info("Saved evaluation report to: %s", md_file)

    return report


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        architecture=args.architecture,
        dataset_path=args.dataset,
        limit=args.limit,
        output_dir=args.output,
        mock=args.mock,
        user_id=args.user_id,
        session_id=args.session_id,
    )
