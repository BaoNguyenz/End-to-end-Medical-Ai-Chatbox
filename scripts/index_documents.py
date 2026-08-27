"""
index_documents.py
Run the full indexing pipeline: load → chunk → embed → store in Qdrant.
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.indexing.document_loader import load_all_documents
from src.indexing.semantic_chunker import SemanticChunker
from src.indexing.vector_store import VectorStore


def main() -> None:
    settings.ensure_dirs()
    total_start = time.time()

    # ------------------------------------------------------------------
    # Step 1: Load documents
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Loading documents")
    print("=" * 60)
    t = time.time()
    docs = load_all_documents(settings.data_dir)
    print(f"  Time: {time.time() - t:.2f}s\n")

    if not docs:
        print("[ERROR] No documents found. Run `python scripts/seed_data.py` first.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: Semantic chunking
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 2: Semantic chunking")
    print("=" * 60)
    t = time.time()
    chunker = SemanticChunker(
        model_name=settings.embedding_model,
        similarity_threshold=settings.chunk_similarity_threshold,
        max_chunk_size=settings.chunk_max_size,
        min_chunk_size=settings.chunk_min_size,
    )
    chunks = chunker.chunk_documents(docs)
    print(f"  Time: {time.time() - t:.2f}s\n")

    # ------------------------------------------------------------------
    # Step 3: Index into Qdrant
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 3: Indexing into Qdrant")
    print("=" * 60)
    t = time.time()
    store = VectorStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=settings.qdrant_collection,
        embedding_model_name=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
        hnsw_m=settings.hnsw_m,
        hnsw_ef_construct=settings.hnsw_ef_construct,
        hnsw_ef_search=settings.hnsw_ef_search,
        _embedding_model=chunker.model,  # reuse loaded model
    )
    store.create_collection(recreate=True)
    store.index_chunks(chunks)
    print(f"  Time: {time.time() - t:.2f}s\n")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_time = time.time() - total_start
    info = store.collection_info()
    avg_chunk_size = sum(len(c.content) for c in chunks) / max(len(chunks), 1)

    print("=" * 60)
    print("INDEXING COMPLETE")
    print("=" * 60)
    print(f"  Documents:     {len(docs)}")
    print(f"  Chunks:        {len(chunks)}")
    print(f"  Avg chunk:     {avg_chunk_size:.0f} chars")
    print(f"  Vectors:       {info['points_count']}")
    print(f"  Collection:    {info['name']} ({info['status']})")
    print(f"  Total time:    {total_time:.2f}s")

    # ------------------------------------------------------------------
    # Quick search test
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("QUICK SEARCH TEST")
    print("=" * 60)
    test_queries = [
        "API authentication",
        "data privacy policy",
        "TechDocs Pro pricing",
    ]
    for q in test_queries:
        results = store.search(q, top_k=3)
        print(f'\n  Query: "{q}"')
        for i, r in enumerate(results):
            print(f"    [{i+1}] score={r.score:.4f} doc={r.chunk.doc_id} | {r.chunk.content[:80]}...")


if __name__ == "__main__":
    main()
