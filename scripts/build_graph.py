"""
build_graph.py
One-time script to extract entities from all documents and populate Neo4j.

Run:
    uv run python scripts/build_graph.py

What it does:
  1. Load all 20 documents from data/
  2. Extract entities (Policy, Stakeholder, Product, Regulation, TechnicalDoc)
     and relationships using OpenAI LLM (cached in cache/entity_cache.json)
  3. Clear existing graph and re-populate with MERGE statements
  4. Print final node/relationship counts
  5. Save extracted entities to cache/graph_entities.json for inspection
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from src.config import settings
from src.indexing.document_loader import load_all_documents
from src.graph.entity_extractor import EntityExtractor
from src.graph.knowledge_graph import KnowledgeGraph

SEP = "=" * 60


def main() -> None:
    settings.ensure_dirs()
    total_start = time.time()

    print(SEP)
    print("STEP 1: Load documents")
    print(SEP)
    docs = load_all_documents(settings.data_dir)
    if not docs:
        print("[ERROR] No documents found. Run seed_data.py first.")
        sys.exit(1)
    print(f"  Loaded {len(docs)} documents\n")

    print(SEP)
    print("STEP 2: Extract entities (LLM + cache)")
    print(SEP)
    client = OpenAI(api_key=settings.openai_api_key)
    extractor = EntityExtractor(
        openai_client=client,
        model=settings.openai_model,
        cache_dir=settings.cache_dir,
    )
    entities = extractor.extract_all(docs)
    print(f"\n  Extraction summary: {entities.summary()}\n")

    # Save extracted entities for manual inspection
    export_path = settings.cache_dir / "graph_entities.json"
    export_path.write_text(
        entities.model_dump_json(indent=2), encoding="utf-8"
    )
    print(f"  Entities saved to: {export_path}\n")

    print(SEP)
    print("STEP 3: Populate Neo4j")
    print(SEP)
    kg = KnowledgeGraph(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    kg.create_indexes()
    kg.clear_graph()

    t = time.time()
    counts = kg.populate(entities)
    print(f"  Time: {time.time()-t:.2f}s\n")

    print(SEP)
    print("BUILD COMPLETE")
    print(SEP)
    for label, count in counts.items():
        print(f"  {label:<15} {count} nodes/relationships")
    print(f"\n  Total time: {time.time()-total_start:.2f}s")
    print()
    print("Next steps:")
    print("  - Neo4j Browser: http://localhost:7474")
    print("  - Run query: MATCH (n) RETURN labels(n)[0], count(*)")
    print("  - Verify: uv run python scripts/test_graph.py")


if __name__ == "__main__":
    main()
