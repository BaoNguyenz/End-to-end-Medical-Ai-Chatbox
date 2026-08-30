"""
evaluate_pipeline.py
Run the full Medical AI RAG pipeline against the 105-question benchmark
and evaluate each response with 5 quality metrics.

Usage:
    uv run python scripts/evaluate_pipeline.py            # full 105 questions
    uv run python scripts/evaluate_pipeline.py --limit 5  # quick dry-run
    uv run python scripts/evaluate_pipeline.py --resume   # resume from checkpoint
    uv run python scripts/evaluate_pipeline.py --no-llm   # skip LLM judge (cost-free)

Output:
    evaluation_report_100.md
    cache/eval_checkpoint.json  (auto-saved after every query)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# ── Bootstrap path ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.orchestrator.pipeline import RAGPipeline
from src.orchestrator.evaluator import Evaluator
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ── Constants ────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# Resolve benchmark path (check Data/ or data/)
if (ROOT_DIR / "Data" / "medical_benchmark_100.json").exists():
    BENCHMARK_PATH = ROOT_DIR / "Data" / "medical_benchmark_100.json"
elif (ROOT_DIR / "data" / "medical_benchmark_100.json").exists():
    BENCHMARK_PATH = ROOT_DIR / "data" / "medical_benchmark_100.json"
else:
    BENCHMARK_PATH = ROOT_DIR / "Data" / "medical_benchmark_100.json"

CHECKPOINT_PATH  = ROOT_DIR / "cache" / "eval_checkpoint.json"
REPORT_PATH      = ROOT_DIR / "evaluation_report_100.md"
SEP = "=" * 70


def load_benchmark() -> list[dict]:
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"completed_ids": [], "results": []}


def save_checkpoint(checkpoint: dict) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)


def safe_print(text: str) -> None:
    """Print text safely on Windows (strips non-ASCII if needed)."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Medical AI RAG Benchmark Evaluator")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to N questions (for quick dry-run)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint (skip already-completed questions)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM-based faithfulness and relevance metrics (saves cost)")
    parser.add_argument("--no-graph", action="store_true",
                        help="Run pipeline without Neo4j graph retrieval")
    args = parser.parse_args()

    run_llm_metrics = not args.no_llm
    use_graph = not args.no_graph

    print(SEP)
    print("  MEDICAL AI RAG BENCHMARK EVALUATOR")
    print(SEP)
    print(f"  Benchmark   : {BENCHMARK_PATH}")
    print(f"  LLM Metrics : {'YES (faithfulness + relevance will be scored)' if run_llm_metrics else 'NO (cost-free mode)'}")
    print(f"  Graph RAG   : {'YES' if use_graph else 'NO'}")
    print(f"  Resume      : {'YES' if args.resume else 'NO'}")
    print(SEP + "\n")

    # ── Load benchmark ─────────────────────────────────────────────────
    questions = load_benchmark()
    if args.limit:
        questions = questions[:args.limit]

    # ── Resume checkpoint ──────────────────────────────────────────────
    checkpoint = load_checkpoint() if args.resume else {"completed_ids": [], "results": []}
    completed_ids = set(checkpoint["completed_ids"])
    pending = [q for q in questions if q["id"] not in completed_ids]

    print(f"  Total questions  : {len(questions)}")
    print(f"  Already done     : {len(completed_ids)}")
    print(f"  Remaining        : {len(pending)}\n")

    if not pending:
        print("[INFO] All questions already evaluated. Generating report from checkpoint...")
    else:
        # ── Initialize pipeline ────────────────────────────────────────
        print("[INIT] Loading pipeline components...")
        t0 = time.time()

        embedding_model = SentenceTransformer(settings.embedding_model)
        openai_client = OpenAI(api_key=settings.openai_api_key)

        pipeline = RAGPipeline(
            use_graph=use_graph,
        pipeline_order="rerank_first",
        )

        evaluator = Evaluator(
            embedding_model=embedding_model,
            openai_client=openai_client,
            openai_model=settings.openai_model,
            log_dir=Path("cache"),
        )

        # Pre-load existing log into evaluator from checkpoint
        evaluator._log = checkpoint.get("results", [])

        print(f"[INIT] Pipeline ready in {time.time()-t0:.1f}s\n")

        # ── Run evaluation loop ────────────────────────────────────────
        for i, q in enumerate(pending, start=len(completed_ids) + 1):
            qid      = q["id"]
            query    = q["query"]
            category = q["category"]
            is_emg   = q.get("is_emergency", False)
            expect_r = q.get("expect_rejection", False)

            safe_print(f"[{i:03d}/{len(questions)}] {qid} | {category:<15} | {query[:65]}")

            try:
                t = time.time()
                response = pipeline.process_query(
                    query,
                    search_mode="auto",
                    top_k=10,
                    use_graph=use_graph,
                )
                elapsed = time.time() - t

                record = evaluator.evaluate(
                    response,
                    query_category=category,
                    is_emergency=is_emg,
                    expect_rejection=expect_r,
                    run_llm_metrics=run_llm_metrics,
                )

                safe_print(
                    f"         ctx={record['context_relevance']:.3f}  "
                    f"faith={record['faithfulness_score']:.3f}  "
                    f"rel={record['relevance_score']:.3f}  "
                    f"safety={record['safety_score']:.1f}  "
                    f"reject={record['rejection_score']:.1f}  "
                    f"({elapsed:.1f}s)"
                )

            except KeyboardInterrupt:
                print("\n[INTERRUPTED] Saving checkpoint before exit...")
                checkpoint["completed_ids"] = list(completed_ids) + [
                    pending[j]["id"] for j in range(i - len(completed_ids))
                ]
                checkpoint["results"] = evaluator._log
                save_checkpoint(checkpoint)
                print(f"[OK] Checkpoint saved to {CHECKPOINT_PATH}")
                sys.exit(0)

            except Exception as e:
                safe_print(f"         [ERROR] {e}")
                record = {
                    "query": query,
                    "category": category,
                    "error": str(e),
                    "context_relevance": 0.0,
                    "faithfulness_score": 0.0,
                    "relevance_score": 0.0,
                    "safety_score": 0.0,
                    "rejection_score": 0.0,
                    "latency": {},
                    "metadata": {},
                    "num_sources": 0,
                    "answer_preview": "",
                }
                evaluator._log.append(record)

            # Save checkpoint after every question
            completed_ids.add(qid)
            checkpoint["completed_ids"] = list(completed_ids)
            checkpoint["results"] = evaluator._log
            save_checkpoint(checkpoint)

    # ── Generate final report ──────────────────────────────────────────────
    print("\n" + SEP)
    print("  GENERATING EVALUATION REPORT")
    print(SEP)

    # Reload evaluator log from full checkpoint if resuming
    if not pending:
        from src.orchestrator.evaluator import Evaluator as Ev
        import numpy as np
        embedding_model = SentenceTransformer(settings.embedding_model)
        openai_client   = OpenAI(api_key=settings.openai_api_key)
        evaluator = Ev(embedding_model, openai_client, settings.openai_model, Path("cache"))
        evaluator._log  = checkpoint.get("results", [])

    report = evaluator.generate_report(output_path=REPORT_PATH)

    # Print summary to console
    log = evaluator._log
    n = len(log)
    if n > 0:
        import numpy as np
        print(f"\n  Total evaluated : {n}")
        print(f"  Context Relevance  : {np.mean([r.get('context_relevance',0) for r in log]):.4f}")
        print(f"  Faithfulness       : {np.mean([r.get('faithfulness_score',0) for r in log]):.4f}")
        print(f"  Answer Relevance   : {np.mean([r.get('relevance_score',0) for r in log]):.4f}")
        print(f"  Medical Safety     : {np.mean([r.get('safety_score',0) for r in log]):.4f}")
        print(f"  Negative Rejection : {np.mean([r.get('rejection_score',0) for r in log]):.4f}")

    print(f"\n  [OK] Report saved to: {REPORT_PATH.resolve()}")
    print(SEP)


if __name__ == "__main__":
    main()
