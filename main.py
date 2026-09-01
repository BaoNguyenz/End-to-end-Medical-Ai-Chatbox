"""
main.py
Interactive CLI for GaleMed AI Clinical Assistant (GraphRAG + Hybrid Search + Redis Cache).

Usage:
    uv run python main.py
    uv run python main.py --no-graph          # disable Neo4j
    uv run python main.py --mode hybrid       # force hybrid search (auto|hybrid|vector|bm25|graph)

Commands during interactive session:
    /quit         Exit
    /mode <mode>  Change search mode (auto|hybrid|vector|bm25|graph)
    /eval         Run full evaluation on benchmark queries and save report
    /cache        Show Redis semantic cache stats
    /clear        Clear Redis semantic cache
    /help         Show commands
"""

import sys
import argparse
import time
from pathlib import Path

# Force UTF-8 encoding on Windows console
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

from src.orchestrator.pipeline import RAGPipeline
from src.orchestrator.evaluator import Evaluator
from src.cache.semantic_cache import get_semantic_cache

BANNER = """
=================================================================
  GaleMed AI Clinical Assistant  |  The Gale Encyclopedia of Medicine
  Components: Hybrid Search + Neo4j GraphRAG + Cross-Encoder + Redis Cache
=================================================================
Type your medical question, or use /help for commands.
"""

HELP_TEXT = """
Available commands:
  /mode <mode>   Set search mode: auto | hybrid | vector | bm25 | graph
  /graph on|off  Toggle Neo4j Knowledge Graph search
  /cache         Display Redis Semantic Cache statistics
  /clear         Clear all Redis Semantic Cache entries
  /eval          Run evaluation benchmark (evaluator.py)
  /help          Show this message
  /quit          Exit the CLI
"""


def parse_args():
    parser = argparse.ArgumentParser(description="GaleMed AI Clinical Assistant CLI")
    parser.add_argument(
        "--mode",
        choices=["auto", "hybrid", "vector", "bm25", "graph"],
        default="auto",
        help="Initial search mode (default: auto)",
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Disable Neo4j Knowledge Graph integration",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of context chunks to pass to LLM (default: 10)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(BANNER)
    print("Initializing pipeline components... (this may take a few seconds)")

    pipeline = RAGPipeline(use_graph=not args.no_graph)
    cache = get_semantic_cache()
    mode = args.mode
    use_graph = not args.no_graph
    top_k = args.top_k

    print(f"\nReady! Search mode: [{mode.upper()}], Graph: [{'ON' if use_graph else 'OFF'}], Cache: [{'ONLINE' if cache.available else 'OFFLINE'}]\n")

    while True:
        try:
            user_input = input("User> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd in ("/quit", "/exit", "/q"):
                print("Goodbye!")
                break
            elif cmd == "/help":
                print(HELP_TEXT)
            elif cmd == "/mode":
                if arg in ("auto", "hybrid", "vector", "bm25", "graph"):
                    mode = arg
                    print(f"[OK] Search mode set to: {mode.upper()}")
                else:
                    print("[ERROR] Invalid mode. Choose: auto | hybrid | vector | bm25 | graph")
            elif cmd == "/graph":
                if arg.lower() in ("on", "true", "1"):
                    use_graph = True
                    print("[OK] Knowledge Graph enabled.")
                elif arg.lower() in ("off", "false", "0"):
                    use_graph = False
                    print("[OK] Knowledge Graph disabled.")
                else:
                    print(f"Graph is currently: {'ON' if use_graph else 'OFF'}. Use: /graph on|off")
            elif cmd == "/cache":
                stats = cache.stats()
                print("\n--- Redis Semantic Cache Stats ---")
                print(f"  Hits            : {stats['hits']}")
                print(f"  Misses          : {stats['misses']}")
                print(f"  Hit Rate        : {stats['hit_rate_pct']}%")
                print(f"  Cached Queries  : {stats['total_cached_queries']}")
                print(f"  Available       : {stats['available']}\n")
            elif cmd == "/clear":
                deleted = cache.clear()
                print(f"[OK] Cleared {deleted} cache entries.")
            elif cmd == "/eval":
                print("\nRunning benchmark evaluation...")
                evaluator = Evaluator(pipeline=pipeline)
                report = evaluator.evaluate()
                print("\nEvaluation Summary:")
                print(report.summary_table())
            else:
                print(f"[ERROR] Unknown command '{cmd}'. Type /help for available commands.")
            continue

        # Process clinical query
        print("\nThinking...", end="", flush=True)
        response = pipeline.process_query(
            query=user_input,
            search_mode=mode,
            top_k=top_k,
            use_graph=use_graph,
        )
        print("\r" + " " * 15 + "\r", end="")  # Clear "Thinking..."

        is_cached = response.metadata.get("cached", False)
        badge = "[CACHED - REDIS]" if is_cached else f"[{response.metadata.get('search_mode', mode).upper()}]"
        
        print(f"\n--- GaleMed AI ({badge}) ---")
        print(response.answer)
        print("\n--- Retrieved Sources ---")
        if not response.sources:
            if is_cached:
                print("  (Served from Redis Semantic Cache in < 10ms)")
            else:
                print("  (No external sources retrieved / Emergency response)")
        else:
            for i, src in enumerate(response.sources[:5]):
                source_type = getattr(src, "source", "hybrid")
                doc_name = src.chunk.doc_id.replace("_", " ").title()
                score_str = f"score={src.score:.3f}" if hasattr(src, "score") else ""
                print(f"  [{i+1}] {doc_name} ({source_type}) {score_str}")

        print(f"\nLatency: {response.latency.get('total', 0)*1000:.0f}ms\n" + "-"*65)


if __name__ == "__main__":
    main()
