"""
build_graph.py
One-time script to extract entities from all documents and populate Neo4j.

Run:
    uv run python scripts/build_graph.py

What it does:
  1. Load all Markdown documents
  2. Build the same canonical chunks used by Qdrant and BM25
  3. Extract entities and chunk-grounded relationships using an LLM
  4. Clear and rebuild Neo4j with evidence_chunk_ids provenance
  5. Save extracted graph data for inspection
"""

import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# pyrefly: ignore [missing-import]
from src.config import settings
from src.llm_client import create_llm_client
# pyrefly: ignore [missing-import]
from src.indexing.document_loader import load_all_documents
from src.indexing.semantic_chunker import SemanticChunker
# pyrefly: ignore [missing-import]
from src.graph.entity_extractor import EntityExtractor
# pyrefly: ignore [missing-import]
from src.graph.knowledge_graph import KnowledgeGraph

SEP = "=" * 60


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a medical graph with local Qwen")
    parser.add_argument("--limit-chunks", type=int, default=None,
                        help="Pilot on the first N canonical chunks; never writes Neo4j")
    parser.add_argument("--extract-only", action="store_true",
                        help="Save extracted graph data without writing Neo4j")
    args = parser.parse_args()
    if args.limit_chunks is not None and args.limit_chunks < 1:
        parser.error("--limit-chunks must be positive")
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
    print("STEP 2: Build canonical chunks")
    print(SEP)
    chunker = SemanticChunker(
        model_name=settings.embedding_model,
        similarity_threshold=settings.chunk_similarity_threshold,
        max_chunk_size=settings.chunk_max_size,
        min_chunk_size=settings.chunk_min_size,
        overlap_sentences=settings.chunk_overlap_sentences,
    )
    chunks = chunker.chunk_documents(docs)
    if args.limit_chunks is not None:
        sample_size = min(args.limit_chunks, len(chunks))
        chunks = [chunks[i * len(chunks) // sample_size] for i in range(sample_size)]
        print(f"  Pilot: selected {len(chunks)} chunks across the corpus; Neo4j will not be changed")

    print(SEP)
    print("STEP 3: Extract chunk-grounded entities (LLM + cache)")
    print(SEP)
    print(f"  Local model: {settings.llm_model} at {settings.llm_base_url}")
    client = create_llm_client()
    extractor = EntityExtractor(
        openai_client=client,
        model=settings.llm_model,
        cache_dir=settings.cache_dir,
    )
    entities = extractor.extract_all_chunks(chunks)
    print(f"\n  Extraction summary: {entities.summary()}\n")

    # Save extracted entities for manual inspection
    export_name = (f"graph_entities_pilot_{len(chunks)}.json"
                   if args.limit_chunks is not None else "graph_entities.json")
    export_path = settings.cache_dir / export_name
    export_path.write_text(
        entities.model_dump_json(indent=2), encoding="utf-8"
    )
    print(f"  Entities saved to: {export_path}\n")

    if args.limit_chunks is not None or args.extract_only:
        if args.limit_chunks is not None:
            review_path = settings.cache_dir / f"graph_pilot_chunks_{len(chunks)}.json"
            review_path.write_text(
                json.dumps(
                    [
                        {"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id,
                         "content": chunk.content}
                        for chunk in chunks
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"  Pilot chunk text saved to: {review_path}")
        print("Extraction complete; Neo4j was not changed.")
        return

    print(SEP)
    print("STEP 4: Populate Neo4j")
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
    print("  - Neo4j Browser: http://localhost:7475")
    print("  - Run query: MATCH (n) RETURN labels(n)[0], count(*)")
    print("  - Verify: uv run python scripts/test_graph.py")


if __name__ == "__main__":
    main()
