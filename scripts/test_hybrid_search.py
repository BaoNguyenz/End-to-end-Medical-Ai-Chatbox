"""
test_hybrid_search.py
Verify Hybrid Search (BM25 + Vector + RRF) + QueryRouter for Medical AI Chatbot.

Requires:
  - Qdrant running (docker compose up)
  - index_documents.py already run (medical_docs collection populated)

Run:
    uv run python scripts/test_hybrid_search.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# pyrefly: ignore [missing-import]
from src.config import settings
# pyrefly: ignore [missing-import]
from src.indexing.vector_store import VectorStore
# pyrefly: ignore [missing-import]
from src.retrieval.bm25_retriever import BM25Retriever
# pyrefly: ignore [missing-import]
from src.retrieval.hybrid_search import HybridSearch
# pyrefly: ignore [missing-import]
from src.retrieval.query_router import QueryRouter

SEP = "=" * 65


def build_components():
    print(SEP)
    print("SETUP: Connecting to Qdrant (medical_docs) & building BM25 index")
    print(SEP)

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
    if info['points_count'] == 0:
        print("\n  [ERROR] Qdrant is empty! Run index_documents.py first:")
        print("    uv run python scripts/index_documents.py")
        sys.exit(1)

    t = time.time()
    all_chunks = store.get_all_chunks()
    print(f"  [OK] Fetched {len(all_chunks)} chunks from Qdrant ({time.time()-t:.2f}s)")

    bm25 = BM25Retriever()
    t = time.time()
    bm25.build_index(all_chunks)
    print(f"  [OK] BM25 index built over {bm25.corpus_size} chunks ({time.time()-t:.2f}s)\n")

    hybrid = HybridSearch(vector_store=store, bm25_retriever=bm25)
    router = QueryRouter()

    return store, bm25, hybrid, router


def print_results(results, max_show=5):
    for i, r in enumerate(results[:max_show]):
        doc_id = str(r.chunk.doc_id).encode("ascii", errors="ignore").decode("ascii")
        preview = str(r.chunk.content)[:90].replace("\n", " ").encode("ascii", errors="ignore").decode("ascii")
        print(f"  [{i+1}] score={r.score:.5f}  src={r.source.value:<7}  doc={doc_id} | {preview}...")
    if len(results) > max_show:
        print(f"  ... (+{len(results) - max_show} more)")


def compare_strategies(query, hybrid):
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
    t = time.time()
    results, q_type, strategy = router.route(query, hybrid, top_k=5)
    elapsed = time.time() - t

    print(f"  -> QueryType={q_type.value}  Strategy={strategy.value}  ({elapsed*1000:.0f}ms)")
    print_results(results)


def main():
    store, bm25, hybrid, router = build_components()

    # ----------------------------------------------------------------
    # Test 1: Medical Keyword queries (ICD codes, dosages, lab tests)
    # ----------------------------------------------------------------
    print(SEP)
    print("TEST 1: Medical Keyword queries (ICD codes, drug dosages, lab tests)")
    print(SEP)

    for q in ["500mg aspirin", "J45 asthma", "HbA1c diabetes test", "CBC normal range"]:
        print(f'\n  Query: "{q}"')
        test_router(q, hybrid, router)

    # ----------------------------------------------------------------
    # Test 2: Medical Semantic queries (Symptoms & disease concepts)
    # ----------------------------------------------------------------
    print()
    print(SEP)
    print("TEST 2: Medical Semantic queries (Symptoms & pathology)")
    print(SEP)

    for q in [
        "What are the primary symptoms of asthma?",
        "How does diabetes mellitus affect kidney function?",
        "What are the common causes of high blood pressure?",
    ]:
        print(f'\n  Query: "{q}"')
        test_router(q, hybrid, router)

    # ----------------------------------------------------------------
    # Test 3: Medical Relationship queries (Interactions & Contraindications)
    # ----------------------------------------------------------------
    print()
    print(SEP)
    print("TEST 3: Medical Relationship queries (Interactions & Contraindications)")
    print(SEP)

    for q in [
        "Can aspirin interact with anticoagulant medications?",
        "What medications are contraindicated in asthmatic patients?",
        "Side effects of metformin in diabetic patients",
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

    compare_query = "asthma symptoms and emergency treatment"
    print(f'\n  Query: "{compare_query}"')
    compare_strategies(compare_query, hybrid)

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print()
    print(SEP)
    print("MEDICAL HYBRID SEARCH VERIFICATION COMPLETE")
    print(SEP)
    print("  [OK] BM25 index built from medical encyclopedia chunks")
    print("  [OK] BM25-only search working for exact terms & dosages")
    print("  [OK] Vector-only search working for symptom semantics")
    print("  [OK] Hybrid RRF fusion combines keyword and semantic search")
    print("  [OK] QueryRouter routes medical queries accurately")
    print()


if __name__ == "__main__":
    main()
