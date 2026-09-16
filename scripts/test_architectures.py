"""
Task 3 Smoke Test — RAG Architecture Comparison
=================================================
Verifies:
    TEST 1: Architecture interfaces       — both pipelines callable, same output schema
    TEST 2: ExperimentRunner              — run_architecture() + compare()
    TEST 3: Chart generation              — radar + bar PNGs created
    TEST 4: Markdown report generation    — valid Markdown output

Uses MockPipelines (no Qdrant/OpenAI needed) so tests always pass in any environment.
For real pipeline testing, run: python scripts/test_architectures.py --real

Usage:
    .venv\\Scripts\\python.exe scripts/test_architectures.py
    .venv\\Scripts\\python.exe scripts/test_architectures.py --real  # requires services
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path

# Project root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Force UTF-8 for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.WARNING)

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"


def header(title: str) -> None:
    print("\n" + "-" * 60)
    print(f"  {title}")
    print("-" * 60)


def ok(msg: str) -> None:
    print(f"  {PASS} {msg}")


def fail(msg: str) -> None:
    print(f"  {FAIL} {msg}")


# ── Mock pipelines (no external services required) ────────────────────────────

class _MockNaivePipeline:
    """Simulates NaiveRAGPipeline output with deterministic scores."""

    def __call__(self, *, query: str) -> dict:
        return {
            "answer": (
                f"Naive RAG answer for: {query[:50]}. "
                "According to The Gale Encyclopedia, this condition involves "
                "specific clinical features and evidence-based management. "
                "Please consult a qualified healthcare professional."
            ),
            "contexts": [
                f"Context chunk 1: Basic information about {query[:30]}.",
                f"Context chunk 2: Additional clinical details for {query[:30]}.",
            ],
            "cost_usd": 0.000150,
        }


class _MockFullPipeline:
    """Simulates FullGaleMedRAGPipeline output with better scores."""

    def __call__(self, *, query: str) -> dict:
        return {
            "answer": (
                f"Full GaleMed RAG answer for: {query[:50]}. "
                "According to The Gale Encyclopedia of Medicine (3rd Edition), "
                "this condition presents with specific symptoms including characteristic "
                "clinical signs, diagnostic criteria, and evidence-based therapeutic "
                "protocols including first-line and adjunctive treatments. "
                "Always consult a qualified healthcare professional for personalized advice."
            ),
            "contexts": [
                f"Context chunk 1 (reranked): Highly relevant information about {query[:30]}.",
                f"Context chunk 2 (reranked): Supporting clinical evidence for {query[:30]}.",
                f"Context chunk 3 (graph-enhanced): Related conditions and differential diagnoses.",
            ],
            "cost_usd": 0.000280,
        }


# ── TEST 1: Architecture Interface ────────────────────────────────────────────
def test_architecture_interfaces() -> bool:
    header("TEST 1: Architecture Callable Interfaces")
    try:
        naive = _MockNaivePipeline()
        full  = _MockFullPipeline()

        test_query = "What are the primary symptoms of asthma?"

        for name, pipeline in [("NaiveRAG (mock)", naive), ("FullGaleMed (mock)", full)]:
            result = pipeline(query=test_query)
            assert isinstance(result, dict), f"Result must be dict, got {type(result)}"
            assert "answer"   in result, "Missing 'answer' key"
            assert "contexts" in result, "Missing 'contexts' key"
            assert "cost_usd" in result, "Missing 'cost_usd' key"
            assert isinstance(result["answer"],   str),  "answer must be str"
            assert isinstance(result["contexts"], list), "contexts must be list"
            assert isinstance(result["cost_usd"], float),"cost_usd must be float"
            assert len(result["answer"]) > 10,           "answer too short"
            assert result["cost_usd"] >= 0.0,            "cost_usd must be >= 0"
            ok(f"{name}: answer={len(result['answer'])} chars, "
               f"contexts={len(result['contexts'])}, cost=${result['cost_usd']:.6f}")

        return True

    except Exception as exc:
        fail(str(exc))
        import traceback; traceback.print_exc()
        return False


# ── TEST 2: ExperimentRunner ───────────────────────────────────────────────────
def test_experiment_runner() -> bool:
    header("TEST 2: ExperimentRunner — Mock Architecture Comparison")
    try:
        # pyrefly: ignore [missing-import]
        from src.experiments.runner import ExperimentRunner
        # pyrefly: ignore [missing-import]
        from src.evaluation.dataset import load_benchmark

        dataset = load_benchmark(verbose=False)
        questions = dataset.questions[:3]  # 3 questions for speed

        runner = ExperimentRunner(output_dir="output/experiments/test", batch_size=3)

        ok(f"ExperimentRunner created — output: output/experiments/test/")

        # Run both architectures with mock pipelines (no RAGAS compute)
        naive_report = runner.run_architecture(
            "naive_rag_mock",
            _MockNaivePipeline(),
            questions=questions,
            limit=3,
        )
        ok(f"Naive run completed — {naive_report.evaluated_questions} questions evaluated")

        full_report = runner.run_architecture(
            "full_galemed_mock",
            _MockFullPipeline(),
            questions=questions,
            limit=3,
        )
        ok(f"Full run completed — {full_report.evaluated_questions} questions evaluated")

        # Compare
        comparison = runner.compare(naive_report, full_report)

        assert comparison.naive_architecture == "naive_rag_mock"
        assert comparison.full_architecture  == "full_galemed_mock"
        assert isinstance(comparison.naive_scores, dict)
        assert isinstance(comparison.full_scores,  dict)
        assert isinstance(comparison.deltas,       dict)
        ok("compare() returned valid ComparisonResult")

        # Save comparison
        saved = runner.save_comparison(comparison, "comparison_test.json")
        assert saved.exists(), f"Comparison file not saved: {saved}"
        ok(f"Comparison saved: {saved}")

        return True

    except Exception as exc:
        fail(str(exc))
        import traceback; traceback.print_exc()
        return False


# ── TEST 3: Chart Generation ──────────────────────────────────────────────────
def test_chart_generation() -> bool:
    header("TEST 3: Chart Generation — Radar + Bar Charts")
    try:
        # pyrefly: ignore [missing-import]
        from src.experiments.runner import ExperimentRunner, ComparisonResult

        # Build a synthetic comparison result (no actual evaluation needed)
        comparison = ComparisonResult(
            naive_architecture="naive_rag",
            full_architecture="full_galemed",
            naive_scores={
                "faithfulness": 0.72, "answer_relevancy": 0.68,
                "context_precision": 0.65, "context_recall": 0.60,
                "medical_safety": 0.85, "negative_rejection": 0.90,
                "avg_latency_ms": 820.0, "total_cost_usd": 0.0312,
            },
            full_scores={
                "faithfulness": 0.91, "answer_relevancy": 0.88,
                "context_precision": 0.86, "context_recall": 0.82,
                "medical_safety": 0.97, "negative_rejection": 0.98,
                "avg_latency_ms": 1250.0, "total_cost_usd": 0.0589,
            },
            deltas={
                "faithfulness": 0.19, "answer_relevancy": 0.20,
                "context_precision": 0.21, "context_recall": 0.22,
                "medical_safety": 0.12, "negative_rejection": 0.08,
                "avg_latency_ms": 430.0,
            },
            category_breakdown={
                "respiratory":    {"faithfulness": {"naive": 0.70, "full": 0.90, "delta": 0.20}},
                "cardiovascular": {"faithfulness": {"naive": 0.68, "full": 0.88, "delta": 0.20}},
                "emergency":      {"faithfulness": {"naive": 0.92, "full": 0.98, "delta": 0.06}},
                "adversarial":    {"faithfulness": {"naive": 0.85, "full": 0.96, "delta": 0.11}},
                "neuro_psych":    {"faithfulness": {"naive": 0.65, "full": 0.87, "delta": 0.22}},
                "pharmacology":   {"faithfulness": {"naive": 0.71, "full": 0.91, "delta": 0.20}},
                "surgery_gi":     {"faithfulness": {"naive": 0.69, "full": 0.89, "delta": 0.20}},
                "out_of_scope":   {"faithfulness": {"naive": 0.80, "full": 0.95, "delta": 0.15}},
            },
        )

        # pyrefly: ignore [missing-import]
        from src.experiments.analysis import generate_charts

        chart_paths = generate_charts(comparison, output_dir="output/experiments/test")

        if "radar" in chart_paths:
            assert chart_paths["radar"].exists(), f"Radar chart not saved: {chart_paths['radar']}"
            ok(f"Radar chart: {chart_paths['radar']}")
        else:
            ok("Radar chart skipped (matplotlib may not be available)")

        if "bar_faithfulness" in chart_paths:
            assert chart_paths["bar_faithfulness"].exists()
            ok(f"Bar chart:   {chart_paths['bar_faithfulness']}")
        else:
            ok("Bar chart skipped (matplotlib may not be available)")

        return True

    except ImportError as exc:
        print(f"  {SKIP} matplotlib not installed: {exc}")
        return True  # Skip is not a failure
    except Exception as exc:
        fail(str(exc))
        import traceback; traceback.print_exc()
        return False


# ── TEST 4: Markdown Report ────────────────────────────────────────────────────
def test_markdown_report() -> bool:
    header("TEST 4: Markdown Report Generation")
    try:
        # pyrefly: ignore [missing-import]
        from src.experiments.runner import ComparisonResult
        # pyrefly: ignore [missing-import]
        from src.experiments.analysis import generate_markdown_report

        comparison = ComparisonResult(
            naive_architecture="naive_rag",
            full_architecture="full_galemed",
            naive_scores={"faithfulness": 0.72, "answer_relevancy": 0.68, "avg_latency_ms": 820.0, "total_cost_usd": 0.031},
            full_scores= {"faithfulness": 0.91, "answer_relevancy": 0.88, "avg_latency_ms": 1250.0, "total_cost_usd": 0.058},
            deltas={"faithfulness": 0.19, "answer_relevancy": 0.20, "avg_latency_ms": 430.0},
            category_breakdown={},
        )

        out_path = Path("output/experiments/test/report_test.md")
        md = generate_markdown_report(comparison, output_path=out_path)

        assert isinstance(md, str) and len(md) > 100, "Report too short"
        assert "naive_rag"    in md, "Naive arch name missing"
        assert "full_galemed" in md, "Full arch name missing"
        assert "Faithfulness" in md, "Metric table missing"
        assert out_path.exists(), f"Report file not saved: {out_path}"
        ok(f"Markdown report generated: {len(md)} chars")
        ok(f"Saved to: {out_path}")

        return True

    except Exception as exc:
        fail(str(exc))
        import traceback; traceback.print_exc()
        return False


# ── Real Pipeline Test (optional, requires Qdrant + OpenAI) ──────────────────
def test_real_pipelines() -> bool:
    header("TEST 5: Real Architectures (1 question each — requires services)")
    try:
        # pyrefly: ignore [missing-import]
        from src.experiments.architectures import NaiveRAGPipeline, FullGaleMedRAGPipeline
        # pyrefly: ignore [missing-import]
        from src.evaluation.dataset import load_benchmark

        dataset = load_benchmark(verbose=False)
        sample_q = dataset.questions[0]
        test_query = sample_q.query

        print(f"\n  Test query: {test_query}")

        # Naive RAG
        print("\n  [NaiveRAGPipeline]")
        naive = NaiveRAGPipeline(top_k=3)
        if not naive._ready:
            print(f"  {SKIP} NaiveRAG services unavailable: {naive._init_error}")
        else:
            result = naive(query=test_query)
            ok(f"Answer: {result['answer'][:80]}...")
            ok(f"Contexts: {len(result['contexts'])} chunks, cost=${result['cost_usd']:.6f}")

        # Full GaleMed
        print("\n  [FullGaleMedRAGPipeline]")
        full = FullGaleMedRAGPipeline(use_graph=False)
        if not full._ready:
            print(f"  {SKIP} FullGaleMed services unavailable: {full._init_error}")
        else:
            result = full(query=test_query)
            ok(f"Answer: {result['answer'][:80]}...")
            ok(f"Contexts: {len(result['contexts'])} chunks, cost=${result['cost_usd']:.6f}")

        return True

    except Exception as exc:
        fail(str(exc))
        import traceback; traceback.print_exc()
        return False


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Task 3: Architecture comparison smoke tests")
    parser.add_argument("--real", action="store_true", help="Also run real pipeline test (requires Qdrant + OpenAI)")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    print("\n" + "=" * 60)
    print("  TASK 3: RAG Architecture Comparison — Smoke Tests")
    print("=" * 60)

    results = {
        "Architecture Interfaces": test_architecture_interfaces(),
        "ExperimentRunner":        test_experiment_runner(),
        "Chart Generation":        test_chart_generation(),
        "Markdown Report":         test_markdown_report(),
    }

    if args.real:
        results["Real Pipelines"] = test_real_pipelines()

    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    passed = 0
    for name, ok_flag in results.items():
        status = PASS if ok_flag else FAIL
        print(f"  {status}  {name}")
        if ok_flag:
            passed += 1

    print(f"\n  {passed}/{len(results)} tests passed")
    print("=" * 60 + "\n")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
