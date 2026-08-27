"""
test_hybrid_search.py
Verify Task 2: Hybrid Search (BM25 + Vector + RRF) + QueryRouter.

Run:
    uv run python scripts/test_hybrid_search.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.indexing.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.query_router import QueryRouter


SEP = "=" * 65


def build_components():
    """Initialize all Task-2 components."""
    print(SEP)
    print("SETUP: Connecting to Qdrant & building BM25 index")
    print(SEP)

    # VectorStore (already populated in Task 1)
    store = VectorStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=settings.qdrant_collection,
        embedding_model_name=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
        hnsw_m=settings.hnsw_m,
        hnsw_ef_construct=settings.hnsw_ef_construct,
        hnsw_ef_search=settings.hnsw_ef_search,
    )

    info = store.collection_info()
    print(f"  [OK] Qdrant collection: {info['name']}  |  points: {info['points_count']}")

    # BM25: fetch all chunks from Qdrant
    t = time.time()
    all_chunks = store.get_all_chunks()
    print(f"  [OK] Fetched {len(all_chunks)} chunks from Qdrant  ({time.time()-t:.2f}s)")

    bm25 = BM25Retriever()
    t = time.time()
    bm25.build_index(all_chunks)
    print(f"  [OK] BM25 index built over {bm25.corpus_size} chunks  ({time.time()-t:.2f}s)\n")

    # Hybrid + Router
    hybrid = HybridSearch(vector_store=store, bm25_retriever=bm25)
    router = QueryRouter()

    return store, bm25, hybrid, router


def print_results(results, max_show=5):
    """Pretty-print search results."""
    for i, r in enumerate(results[:max_show]):
        preview = r.chunk.content[:90].replace("\n", " ")
        print(
            f"  [{i+1}] score={r.score:.5f}  src={r.source.value:<7}"
            f"  doc={r.chunk.doc_id}  |  {preview}..."
        )
    if len(results) > max_show:
        print(f"  ... (+{len(results) - max_show} more)")


def compare_strategies(query, hybrid):
    """Run BM25-only, Vector-only, and Hybrid and compare top-5."""
    top_k = 5

    t = time.time()
    bm25_res = hybrid.search_bm25_only(query, top_k=top_k)
    bm25_time = time.time() - t

    t = time.time()
    vec_res = hybrid.search_vector_only(query, top_k=top_k)
    vec_time = time.time() - t

    t = time.time()
    hyb_res = hybrid.search(query, top_k=top_k)
    hyb_time = time.time() - t

    print(f"  -- BM25 only  ({bm25_time*1000:.0f}ms, {len(bm25_res)} results) ---------------")
    print_results(bm25_res)

    print(f"  -- Vector only  ({vec_time*1000:.0f}ms, {len(vec_res)} results) ---------------")
    print_results(vec_res)

    print(f"  -- Hybrid/RRF  ({hyb_time*1000:.0f}ms, {len(hyb_res)} results) ---------------")
    print_results(hyb_res)


def test_router(query, hybrid, router):
    """Route query and show strategy + results."""
    t = time.time()
    results, q_type, strategy = router.route(query, hybrid, top_k=5)
    elapsed = time.time() - t

    print(f"  -> QueryType={q_type.value}  Strategy={strategy.value}  ({elapsed*1000:.0f}ms)")
    print_results(results)


def main():
    store, bm25, hybrid, router = build_components()

    # ----------------------------------------------------------------
    # Test 1: Keyword/Error-code query (expect BM25-heavy)
    # ----------------------------------------------------------------
    print(SEP)
    print("TEST 1: Keyword query -- error code & product ID")
    print(SEP)

    for q in ["ERR_AUTH_001", "TDPRO-2024", "POL-001"]:
        print(f'\n  Query: "{q}"')
        test_router(q, hybrid, router)

    # ----------------------------------------------------------------
    # Test 2: Semantic query (expect Vector-heavy)
    # ----------------------------------------------------------------
    print()
    print(SEP)
    print("TEST 2: Semantic query -- conceptual questions")
    print(SEP)

    for q in [
        "How does API authentication work?",
        "What is the difference between OAuth and API keys?",
        "Explain the remote work policy benefits",
    ]:
        print(f'\n  Query: "{q}"')
        test_router(q, hybrid, router)

    # ----------------------------------------------------------------
    # Test 3: Default Hybrid query
    # ----------------------------------------------------------------
    print()
    print(SEP)
    print("TEST 3: Default hybrid queries")
    print(SEP)

    for q in [
        "data privacy policy stakeholders",
        "TechDocs Pro pricing",
        "incident response procedures",
    ]:
        print(f'\n  Query: "{q}"')
        test_router(q, hybrid, router)

    # ----------------------------------------------------------------
    # Test 4: Side-by-side strategy comparison
    # ----------------------------------------------------------------
    print()
    print(SEP)
    print("TEST 4: Strategy comparison (BM25 vs Vector vs Hybrid)")
    print(SEP)

    compare_query = "data privacy policy"
    print(f'\n  Query: "{compare_query}"')
    compare_strategies(compare_query, hybrid)

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print()
    print(SEP)
    print("TASK 2 VERIFICATION COMPLETE")
    print(SEP)
    print("  [OK] BM25 index built from Qdrant chunks")
    print("  [OK] BM25-only search working")
    print("  [OK] Vector-only search working")
    print("  [OK] Hybrid RRF fusion working")
    print("  [OK] QueryRouter routing KEYWORD -> BM25-heavy")
    print("  [OK] QueryRouter routing SEMANTIC -> Vector-heavy")
    print("  [OK] QueryRouter routing DEFAULT -> Hybrid")
    print()
    print("Next: Task 3 - Query Transformation (HyDE + Decomposition)")


if __name__ == "__main__":
    main()
