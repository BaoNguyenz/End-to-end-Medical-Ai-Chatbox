"""
GaleMed AI — Experiment Runner
================================
Orchestrates RAGAS evaluation across multiple pipeline architectures
on the 105-question medical benchmark.

Usage:
    runner = ExperimentRunner(output_dir="output/experiments")
    naive_report   = runner.run_architecture("naive_rag",    NaiveRAGPipeline())
    full_report    = runner.run_architecture("full_galemed", FullGaleMedRAGPipeline())
    comparison     = runner.compare(naive_report, full_report)
    runner.save_comparison(comparison)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """
    Side-by-side comparison of two architecture evaluation reports.

    Stores the delta (Full - Naive) for each metric so that
    positive values always mean "Full GaleMed improved over Naive RAG".
    """

    naive_architecture: str
    full_architecture: str

    # Aggregate scores per architecture
    naive_scores: dict = field(default_factory=dict)
    full_scores: dict  = field(default_factory=dict)
    deltas: dict       = field(default_factory=dict)   # full - naive

    # Per-category breakdown (nested: {category: {metric: {arch: score}}})
    category_breakdown: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "naive_architecture": self.naive_architecture,
            "full_architecture":  self.full_architecture,
            "naive_scores":  self.naive_scores,
            "full_scores":   self.full_scores,
            "deltas":        self.deltas,
            "category_breakdown": self.category_breakdown,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("Comparison saved to: %s", path)

    def print_summary(self) -> None:
        """Print a formatted comparison table to stdout."""
        metrics = [
            ("Faithfulness",       "faithfulness"),
            ("Answer Relevancy",   "answer_relevancy"),
            ("Context Precision",  "context_precision"),
            ("Context Recall",     "context_recall"),
            ("Medical Safety",     "medical_safety"),
            ("Negative Rejection", "negative_rejection"),
        ]

        def _f(v) -> str:
            return f"{v:.4f}" if v is not None else "  N/A  "

        def _delta(v) -> str:
            if v is None:
                return "   N/A  "
            sign = "+" if v >= 0 else ""
            return f"{sign}{v:.4f}"

        print("\n" + "=" * 72)
        print("  Architecture Comparison: Naive RAG vs Full GaleMed")
        print("=" * 72)
        print(f"  {'Metric':<22} {'Naive RAG':>10} {'Full GaleMed':>13} {'Delta':>10}")
        print("-" * 72)

        for label, key in metrics:
            n = self.naive_scores.get(key)
            f = self.full_scores.get(key)
            d = self.deltas.get(key)
            print(f"  {label:<22} {_f(n):>10} {_f(f):>13} {_delta(d):>10}")

        print("-" * 72)
        n_lat = self.naive_scores.get("avg_latency_ms")
        f_lat = self.full_scores.get("avg_latency_ms")
        d_lat = self.deltas.get("avg_latency_ms")
        print(f"  {'Avg Latency (ms)':<22} {_f(n_lat):>10} {_f(f_lat):>13} {_delta(d_lat):>10}")

        n_cost = self.naive_scores.get("total_cost_usd", 0.0)
        f_cost = self.full_scores.get("total_cost_usd", 0.0)
        print(f"  {'Total Cost (USD)':<22} ${n_cost:.4f}    ${f_cost:.4f}")
        print("=" * 72 + "\n")


class ExperimentRunner:
    """
    Orchestrates evaluation of pipeline architectures using RagasEvaluator.

    Args:
        output_dir:  Directory to save reports (default: output/experiments).
        batch_size:  Questions per RAGAS batch (default: 5).
        resume:      Resume from checkpoint if available.
    """

    def __init__(
        self,
        output_dir: str | Path = "output/experiments",
        batch_size: int = 5,
        resume: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.resume = resume

    def run_architecture(
        self,
        architecture_name: str,
        pipeline,
        questions=None,
        limit: Optional[int] = None,
    ):
        """
        Run RAGAS evaluation for one architecture on the benchmark.

        Args:
            architecture_name: Label for this run (e.g., "naive_rag").
            pipeline:  Callable with signature (query=str) -> dict.
            questions: List of BenchmarkQuestion. If None, loads full 105-Q dataset.
            limit:     Cap the number of questions (for smoke testing).

        Returns:
            EvaluationReport with aggregated metrics.
        """
        # pyrefly: ignore [missing-import]
        from src.evaluation.dataset import load_benchmark
        # pyrefly: ignore [missing-import]
        from src.evaluation.ragas_evaluator import RagasEvaluator

        if questions is None:
            dataset = load_benchmark(verbose=False)
            questions = dataset.questions

        checkpoint_path = self.output_dir / f"checkpoint_{architecture_name}.json"
        evaluator = RagasEvaluator(
            pipeline=pipeline,
            architecture=architecture_name,
            checkpoint_path=checkpoint_path,
            batch_size=self.batch_size,
        )

        logger.info(
            "[Runner] Starting '%s' evaluation on %d questions (limit=%s)...",
            architecture_name, len(questions), limit,
        )
        t0 = time.time()

        report = evaluator.run(
            questions=questions,
            limit=limit,
            resume=self.resume,
            save_results_path=self.output_dir / f"report_{architecture_name}.json",
        )

        elapsed = time.time() - t0
        logger.info(
            "[Runner] '%s' finished in %.1fs — %d/%d evaluated",
            architecture_name, elapsed, report.evaluated_questions, report.total_questions,
        )
        return report

    def compare(self, naive_report, full_report) -> ComparisonResult:
        """
        Build a ComparisonResult from two EvaluationReports.

        Args:
            naive_report: EvaluationReport from NaiveRAGPipeline.
            full_report:  EvaluationReport from FullGaleMedRAGPipeline.

        Returns:
            ComparisonResult with deltas and per-category breakdown.
        """
        metrics = [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "medical_safety",
            "negative_rejection",
            "avg_latency_ms",
        ]

        def _get(report, metric: str):
            return getattr(report, f"avg_{metric}" if metric != "avg_latency_ms" else metric, None)

        naive_scores = {m: _get(naive_report, m) for m in metrics}
        naive_scores["total_cost_usd"] = naive_report.total_cost_usd

        full_scores = {m: _get(full_report, m) for m in metrics}
        full_scores["total_cost_usd"] = full_report.total_cost_usd

        deltas = {}
        for m in metrics:
            n, f = naive_scores[m], full_scores[m]
            deltas[m] = round(f - n, 6) if (n is not None and f is not None) else None

        # Per-category breakdown
        category_breakdown = self._build_category_breakdown(naive_report, full_report, metrics[:-1])

        return ComparisonResult(
            naive_architecture=naive_report.architecture,
            full_architecture=full_report.architecture,
            naive_scores=naive_scores,
            full_scores=full_scores,
            deltas=deltas,
            category_breakdown=category_breakdown,
        )

    def _build_category_breakdown(
        self,
        naive_report,
        full_report,
        metrics: list[str],
    ) -> dict:
        """Build per-category metric averages for both architectures."""
        from collections import defaultdict

        def _group(report) -> dict:
            groups: dict = defaultdict(list)
            for r in report.results:
                groups[r.category].append(r)
            return dict(groups)

        def _avg(results_list, metric: str):
            vals = [getattr(r, metric) for r in results_list if getattr(r, metric) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        naive_groups = _group(naive_report)
        full_groups  = _group(full_report)
        all_cats = sorted(set(list(naive_groups.keys()) + list(full_groups.keys())))

        breakdown: dict = {}
        for cat in all_cats:
            breakdown[cat] = {}
            for m in metrics:
                n_avg = _avg(naive_groups.get(cat, []), m)
                f_avg = _avg(full_groups.get(cat, []), m)
                breakdown[cat][m] = {
                    "naive": n_avg,
                    "full":  f_avg,
                    "delta": round(f_avg - n_avg, 4) if (n_avg is not None and f_avg is not None) else None,
                }

        return breakdown

    def save_comparison(self, comparison: ComparisonResult, filename: str = "comparison.json") -> Path:
        """Save comparison to output_dir/filename."""
        out_path = self.output_dir / filename
        comparison.save(out_path)
        return out_path
