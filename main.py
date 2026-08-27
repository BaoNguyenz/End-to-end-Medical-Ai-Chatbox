"""
main.py
Interactive CLI for the Enterprise RAG system.

Usage:
    uv run python main.py
    uv run python main.py --no-graph          # disable Neo4j
    uv run python main.py --mode hybrid       # force hybrid search

Commands during interactive session:
    /quit         Exit
    /mode <mode>  Change search mode (auto|hybrid|vector|bm25|graph)
    /eval         Run full evaluation on sample queries and save report
    /help         Show commands
"""

import sys
import argparse
import time
from pathlib import Path

from src.orchestrator.pipeline import RAGPipeline
from src.orchestrator.evaluator import Evaluator

BANNER = """
=================================================================
  Enterprise RAG System  |  TechDocs Inc.
  Components: Hybrid Search + GraphRAG + Cross-Encoder + MMR
=================================================================
Type your question, or use /help for commands.
"""

SAMPLE_QUERIES = [
    "How does API authentication work?",
    "What are the data privacy GDPR requirements?",
    "Who is responsible for incident response?",
    "Compare data privacy and information security policies",
    "What is the price of TechDocs Pro?",
    "ERR_AUTH_001",
]


def print_response(response) -> None:
    """Pretty-print a RAGResponse to the terminal."""
    print()
    print("-" * 65)
    print("ANSWER:")
    print(response.answer)
    print()
    print("SOURCES:")
    for i, r in enumerate(response.sources[:5]):
        preview = r.chunk.content[:80].replace("\n", " ")
        print(f"  [{i+1}] {r.chunk.doc_id}  (score={r.score:.3f})  |  {preview}...")
    print()
    print("LATENCY:")
    for stage, t in response.latency.items():
        if stage != "total":
            print(f"  {stage:<20} {t*1000:.0f}ms")
    print(f"  {'TOTAL':<20} {response.latency.get('total', 0)*1000:.0f}ms")
    print(f"\n  mode={response.metadata.get('search_mode')} | "
          f"class={response.metadata.get('query_class')} | "
          f"transform={response.metadata.get('transformation')} | "
          f"candidates={response.metadata.get('num_candidates')}")
    print("-" * 65)


def run_eval(pipeline: RAGPipeline) -> None:
    """Run evaluation on sample queries and save evaluation_report.md."""
    evaluator = Evaluator(
        embedding_model=pipeline._embedding_model,
        openai_client=pipeline.openai_client,
        openai_model=pipeline.openai_client.models.list().data[0].id if False else "gpt-4o-mini",
    )

    from src.config import settings
    # Override model from settings
    evaluator.model = settings.openai_model

    print("\nRunning evaluation on sample queries...")
    for i, q in enumerate(SAMPLE_QUERIES):
        print(f"\n[{i+1}/{len(SAMPLE_QUERIES)}] {q}")
        try:
            resp = pipeline.process_query(q, top_k=5)
            metrics = evaluator.evaluate(resp)
            print(f"  ctx_rel={metrics['context_relevance']:.3f}  "
                  f"faith={metrics['faithfulness_score']:.3f}  "
                  f"lat={resp.latency.get('total',0):.2f}s")
        except Exception as e:
            print(f"  ERROR: {e}")

    report_path = Path("evaluation_report.md")
    evaluator.generate_report(output_path=report_path)
    print(f"\nEvaluation report saved to: {report_path.absolute()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enterprise RAG CLI")
    parser.add_argument("--no-graph", action="store_true", help="Disable Neo4j graph retrieval")
    parser.add_argument("--mode", default="auto",
                        choices=["auto", "hybrid", "vector", "bm25", "graph"],
                        help="Default search mode")
    args = parser.parse_args()

    print("Initializing RAG pipeline (this may take 30-60 seconds)...")
    pipeline = RAGPipeline(use_graph=not args.no_graph)
    current_mode = args.mode

    print(BANNER)
    print(f"Search mode: {current_mode} (change with /mode <mode>)")
    print(f"Graph:       {'enabled' if pipeline.graph_retriever else 'disabled'}\n")

    while True:
        try:
            query = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not query:
            continue

        if query.startswith("/"):
            cmd = query.split()
            if cmd[0] == "/quit":
                print("Bye!")
                break
            elif cmd[0] == "/mode" and len(cmd) > 1:
                current_mode = cmd[1]
                print(f"Mode changed to: {current_mode}")
            elif cmd[0] == "/eval":
                run_eval(pipeline)
            elif cmd[0] == "/help":
                print("Commands: /quit  /mode <auto|hybrid|vector|bm25|graph>  /eval  /help")
            else:
                print(f"Unknown command: {cmd[0]}")
            continue

        try:
            response = pipeline.process_query(query, search_mode=current_mode)
            print_response(response)
        except Exception as e:
            print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()
