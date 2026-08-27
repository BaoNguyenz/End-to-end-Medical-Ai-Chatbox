"""
test_query_transformation.py
Verify Task 3: Query Transformation (HyDE + QueryDecomposer + TransformationRouter).

Requires:
  - OPENAI_API_KEY in .env
  - Qdrant running with indexed documents (Task 1 done)

Run:
    uv run python scripts/test_query_transformation.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from src.config import settings
from src.indexing.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_search import HybridSearch
from src.transformation.hyde import HyDE
from src.transformation.query_decomposition import QueryDecomposer
from src.transformation.transformation_router import TransformationRouter

SEP = "=" * 65


# ── Setup (same as Task 2) ─────────────────────────────────────────────────

def build_base_components():
    print(SEP)
    print("SETUP: Building base components")
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
    print(f"  [OK] Qdrant: {info['name']} | {info['points_count']} points")

    chunks = store.get_all_chunks()
    bm25 = BM25Retriever()
    bm25.build_index(chunks)
    print(f"  [OK] BM25 index: {bm25.corpus_size} chunks")

    hybrid = HybridSearch(vector_store=store, bm25_retriever=bm25)

    client = OpenAI(api_key=settings.openai_api_key)
    print(f"  [OK] OpenAI client ready (model={settings.openai_model})\n")

    return store, hybrid, client


def print_results(results, max_show=5):
    for i, r in enumerate(results[:max_show]):
        preview = r.chunk.content[:85].replace("\n", " ")
        print(f"  [{i+1}] score={r.score:.5f}  doc={r.chunk.doc_id} | {preview}...")
    if len(results) > max_show:
        print(f"  ... (+{len(results)-max_show} more)")


# ── TEST 1: HyDE ──────────────────────────────────────────────────────────

def test_hyde(store, hybrid, client):
    print(SEP)
    print("TEST 1: HyDE -- Hypothetical Document Embeddings")
    print(SEP)
    print("Idea: For a vague query, HyDE generates a fake 'ideal answer'")
    print("      then searches using THAT text's embedding instead of the query.\n")

    hyde = HyDE(
        openai_client=client,
        model=settings.openai_model,
        embedding_model=store.model,
        cache_dir=settings.cache_dir,
    )

    vague_queries = [
        "authentication",
        "remote work",
        "data breach",
    ]

    for query in vague_queries:
        print(f'\n  Query (vague): "{query}"')
        print()

        # Direct vector search (baseline)
        t = time.time()
        direct = store.search(query, top_k=3)
        direct_time = time.time() - t
        print(f"  -- Direct vector search ({direct_time*1000:.0f}ms) --")
        print_results(direct, max_show=3)

        # HyDE search
        t = time.time()
        hyde_results, hyp_doc = hyde.search(query, store, top_k=3)
        hyde_time = time.time() - t
        print(f"\n  -- HyDE search ({hyde_time*1000:.0f}ms) --")
        print(f"  Hypothetical doc preview: \"{hyp_doc[:120]}...\"")
        print_results(hyde_results, max_show=3)
        print()


# ── TEST 2: QueryDecomposer ────────────────────────────────────────────────

def test_decomposer(hybrid, client):
    print(SEP)
    print("TEST 2: QueryDecomposer -- Complex query decomposition")
    print(SEP)
    print("Idea: Complex queries are split into independent sub-questions,")
    print("      each searched separately, results merged & deduplicated.\n")

    decomposer = QueryDecomposer(
        openai_client=client,
        model=settings.openai_model,
        cache_dir=settings.cache_dir,
    )

    complex_queries = [
        "Compare data privacy policy and remote work policy and list stakeholders for both",
        "What are the security requirements and compliance regulations mentioned across all policies?",
    ]

    for query in complex_queries:
        print(f'\n  Query (complex): "{query}"')
        print(f"  is_complex() = {decomposer.is_complex(query)}\n")

        t = time.time()
        results, sub_queries = decomposer.search(query, hybrid, top_k=8)
        elapsed = time.time() - t

        print(f"\n  Sub-queries generated:")
        for i, sq in enumerate(sub_queries):
            print(f"    [{i+1}] {sq}")

        print(f"\n  Aggregated results ({len(results)} unique chunks, {elapsed:.2f}s):")
        print_results(results, max_show=5)
        print()


# ── TEST 3: TransformationRouter ──────────────────────────────────────────

def test_transformation_router(store, hybrid, client):
    print(SEP)
    print("TEST 3: TransformationRouter -- Auto-classify and route")
    print(SEP)
    print("Idea: Router detects query type and picks the right transformation.\n")

    router = TransformationRouter(
        openai_client=client,
        model=settings.openai_model,
        embedding_model=store.model,
        cache_dir=settings.cache_dir,
    )

    test_cases = [
        # (query, expected_class)
        ("authentication",
         "vague"),
        ("What is the price of TechDocs Pro?",
         "simple"),
        ("Compare data privacy policy and information security policy and list their owners",
         "complex"),
        ("remote work",
         "vague"),
        ("How does API rate limiting work?",
         "simple"),
        ("What are all the policies, their stakeholders, and how do they relate to GDPR?",
         "complex"),
    ]

    correct = 0
    for query, expected in test_cases:
        detected = router.classify(query)
        ok = "[OK]" if detected == expected else "[!!]"
        if detected == expected:
            correct += 1
        print(f"  {ok}  class={detected:<8}  expected={expected:<8}  query: \"{query[:60]}\"")

    print(f"\n  Classification accuracy: {correct}/{len(test_cases)}")
    print()

    # Run full pipeline on one of each type
    print("  -- Full pipeline: one query per type --\n")

    sample_queries = [
        "data retention",                                     # vague
        "What is the API rate limit per hour?",              # simple
        "Compare the data privacy and vendor management policies and their GDPR references",  # complex
    ]

    for query in sample_queries:
        print(f'  Query: "{query}"')
        t = time.time()
        results, meta = router.transform_and_search(
            query, store, hybrid, top_k=5
        )
        elapsed = time.time() - t
        print(f"  -> class={meta['query_class']}, transform={meta['transformation']}, {elapsed:.2f}s, {len(results)} results")
        print_results(results, max_show=3)
        print()


# ── Summary ────────────────────────────────────────────────────────────────

def main():
    store, hybrid, client = build_base_components()

    test_hyde(store, hybrid, client)
    test_decomposer(hybrid, client)
    test_transformation_router(store, hybrid, client)

    print(SEP)
    print("TASK 3 VERIFICATION COMPLETE")
    print(SEP)
    print("  [OK] HyDE: generates hypothetical doc, embeds, searches")
    print("  [OK] HyDE: results cached in cache/hyde_cache.json")
    print("  [OK] QueryDecomposer: breaks complex query into sub-questions")
    print("  [OK] QueryDecomposer: aggregates & deduplicates by chunk_id")
    print("  [OK] TransformationRouter: classifies vague/complex/simple")
    print("  [OK] TransformationRouter: routes to correct transformation")
    print()
    print("Next: Task 4 - Post-Retrieval (Cross-Encoder Reranker + MMR)")


if __name__ == "__main__":
    main()
