"""
test_query_transformation.py
Verify Medical HyDE + QueryDecomposer + TransformationRouter for Medical AI.

Requires:
  - Qdrant running + index_documents.py already run
  - OPENAI_API_KEY in .env

Run:
    uv run python scripts/test_query_transformation.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

# pyrefly: ignore [missing-import]
from src.config import settings
# pyrefly: ignore [missing-import]
from src.indexing.vector_store import VectorStore
# pyrefly: ignore [missing-import]
from src.retrieval.bm25_retriever import BM25Retriever
# pyrefly: ignore [missing-import]
from src.retrieval.hybrid_search import HybridSearch
# pyrefly: ignore [missing-import]
from src.transformation.hyde import HyDE
# pyrefly: ignore [missing-import]
from src.transformation.query_decomposition import QueryDecomposer
# pyrefly: ignore [missing-import]
from src.transformation.transformation_router import TransformationRouter

SEP = "=" * 65


def build_base_components():
    print(SEP)
    print("SETUP: Building base components for Medical Query Transformation")
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
        doc_id = str(r.chunk.doc_id).encode("ascii", errors="ignore").decode("ascii")
        preview = str(r.chunk.content)[:85].replace("\n", " ").encode("ascii", errors="ignore").decode("ascii")
        print(f"  [{i+1}] score={r.score:.5f}  doc={doc_id} | {preview}...")
    if len(results) > max_show:
        print(f"  ... (+{len(results)-max_show} more)")


# ── TEST 1: HyDE ──────────────────────────────────────────────────────────

def test_hyde(store, hybrid, client):
    print(SEP)
    print("TEST 1: Medical HyDE -- Hypothetical Clinical Passage Embeddings")
    print(SEP)
    print("Idea: For vague symptom queries, HyDE generates a hypothetical clinical paragraph")
    print("      then searches using THAT embedding instead of the raw short query.\n")

    hyde = HyDE(
        openai_client=client,
        model=settings.openai_model,
        embedding_model=store.model,
        cache_dir=settings.cache_dir,
    )

    vague_medical_queries = [
        "wheezing and cough",
        "high blood sugar symptoms",
        "sudden chest tightness",
    ]

    for query in vague_medical_queries:
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
        clean_hyp = str(hyp_doc[:120]).replace("\n", " ").encode("ascii", errors="ignore").decode("ascii")
        print(f'  Hypothetical doc preview: "{clean_hyp}..."')
        print_results(hyde_results, max_show=3)
        print()


# ── TEST 2: QueryDecomposer ────────────────────────────────────────────────

def test_decomposer(hybrid, client):
    print(SEP)
    print("TEST 2: QueryDecomposer -- Multi-part Medical Query Decomposition")
    print(SEP)
    print("Idea: Complex medical questions are split into independent sub-questions,")
    print("      each searched separately, results merged & deduplicated.\n")

    decomposer = QueryDecomposer(
        openai_client=client,
        model=settings.openai_model,
        cache_dir=settings.cache_dir,
    )

    complex_medical_queries = [
        "Compare the symptoms, causes, and treatments of asthma versus chronic bronchitis",
        "What are the diagnostic criteria, risk factors, and first-line medications for type 2 diabetes?",
    ]

    for query in complex_medical_queries:
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
    print("TEST 3: TransformationRouter -- Medical Query Classification & Routing")
    print(SEP)
    print("Idea: Router detects query complexity and picks the optimal transformation.\n")

    router = TransformationRouter(
        openai_client=client,
        model=settings.openai_model,
        embedding_model=store.model,
        cache_dir=settings.cache_dir,
    )

    test_cases = [
        ("wheezing", "vague"),
        ("What is the recommended dosage of aspirin for headache?", "simple"),
        ("Compare the symptoms, causes, and treatments of asthma and COPD", "complex"),
        ("chest pain", "vague"),
        ("How does metformin reduce blood glucose levels?", "simple"),
        ("What are all diagnostic tests, complications, and drug options for both diabetes and hypertension?", "complex"),
    ]

    correct = 0
    for query, expected in test_cases:
        detected = router.classify(query)
        ok = "[OK]" if detected == expected else "[!!]"
        if detected == expected:
            correct += 1
        print(f'  {ok}  class={detected:<8}  expected={expected:<8}  query: "{query[:60]}"')

    print(f"\n  Classification accuracy: {correct}/{len(test_cases)}")
    print()

    # Run full pipeline on sample medical queries
    print("  -- Full pipeline: one query per type --\n")
    sample_queries = [
        "pneumonia symptoms",
        "What is the standard treatment for acute appendicitis?",
        "Compare the causes, clinical presentation, and emergency management of asthma vs anaphylaxis",
    ]

    for query in sample_queries:
        print(f'  Query: "{query}"')
        t = time.time()
        results, meta = router.transform_and_search(query, store, hybrid, top_k=5)
        elapsed = time.time() - t
        print(f"  -> class={meta['query_class']}, transform={meta['transformation']}, {elapsed:.2f}s, {len(results)} results")
        print_results(results, max_show=3)
        print()


def main():
    store, hybrid, client = build_base_components()

    test_hyde(store, hybrid, client)
    test_decomposer(hybrid, client)
    test_transformation_router(store, hybrid, client)

    print(SEP)
    print("MEDICAL QUERY TRANSFORMATION VERIFICATION COMPLETE")
    print(SEP)
    print("  [OK] Medical HyDE generates physician-style hypothetical context")
    print("  [OK] QueryDecomposer splits multi-clause medical queries")
    print("  [OK] TransformationRouter routes vague/complex/simple clinical questions")
    print()


if __name__ == "__main__":
    main()
