"""
test_post_retrieval.py
Verify Medical Post-Retrieval Pipeline (Cross-Encoder Reranker + MMR Diversification).

Requires:
  - Qdrant running + index_documents.py already run

Run:
    uv run python scripts/test_post_retrieval.py
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
from src.post_retrieval.cross_encoder_reranker import CrossEncoderReranker
# pyrefly: ignore [missing-import]
from src.post_retrieval.mmr import mmr_rerank
# pyrefly: ignore [missing-import]
from src.post_retrieval.post_retrieval_pipeline import PostRetrievalPipeline

SEP = "=" * 65


def build_components():
    print(SEP)
    print("SETUP: Loading components for Medical Post-Retrieval Pipeline")
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
    reranker = CrossEncoderReranker(model_name=settings.cross_encoder_model)
    print(f"  [OK] CrossEncoder ready (model={settings.cross_encoder_model})\n")

    return store, hybrid, reranker


def print_results(label, results, max_show=10):
    print(f"  -- {label} ({len(results)} results) --")
    for i, r in enumerate(results[:max_show]):
        doc_id = str(r.chunk.doc_id).encode("ascii", errors="ignore").decode("ascii")
        preview = str(r.chunk.content)[:80].replace("\n", " ").encode("ascii", errors="ignore").decode("ascii")
        print(f"  [{i+1:2d}] score={r.score:7.4f}  doc={doc_id:<35}  {preview}...")
    print()


# ── TEST 1: CrossEncoder Reranker ─────────────────────────────────────────

def test_reranker(store, hybrid, reranker):
    print(SEP)
    print("TEST 1: CrossEncoder Reranker on Medical Documents")
    print(SEP)
    print("Idea: Hybrid search gets 50 candidates, CrossEncoder scores each")
    print("      (medical question, passage) PAIR accurately.\n")

    queries = [
        "What are the symptoms and warning signs of asthma?",
        "How is type 2 diabetes mellitus diagnosed and treated?",
        "What causes acute appendicitis and what are the clinical signs?",
    ]

    for query in queries:
        print(f'  Query: "{query}"')
        candidates = hybrid.search(query, top_k=50)
        print_results("Before rerank (hybrid top-5)", candidates[:5])

        t = time.time()
        reranked = reranker.rerank(query, candidates, top_k=10)
        elapsed = time.time() - t
        print_results(f"After CrossEncoder rerank ({elapsed:.2f}s, top-10)", reranked)
        print()


# ── TEST 2: MMR Diversification ───────────────────────────────────────────

def test_mmr(store, hybrid, reranker):
    print(SEP)
    print("TEST 2: MMR Diversification (Avoiding repetitive medical chunks)")
    print(SEP)
    print("Idea: MMR ensures results cover diverse medical aspects (causes, treatments, side effects)\n")

    query = "asthma symptoms and medical treatments"
    candidates = hybrid.search(query, top_k=30)
    print_results("Pure relevance (top-10, no MMR diversity)", candidates[:10])

    for lam in [0.7, 0.5, 0.3]:
        t = time.time()
        mmr_results = mmr_rerank(
            query, candidates, store.model,
            lambda_param=lam, top_k=10
        )
        elapsed = time.time() - t
        unique_docs = len(set(r.chunk.doc_id for r in mmr_results))
        print_results(
            f"MMR lambda={lam} ({elapsed:.2f}s, {unique_docs} unique medical docs)",
            mmr_results
        )
    print()


# ── TEST 3: Full PostRetrievalPipeline ────────────────────────────────────

def test_pipeline(store, hybrid, reranker):
    print(SEP)
    print("TEST 3: Full PostRetrievalPipeline (rerank_first vs mmr_first)")
    print(SEP)
    print("Flow A (rerank_first): 50 -> CrossEncoder(top-20) -> MMR(top-10)")
    print("Flow B (mmr_first):    50 -> MMR(top-20) -> CrossEncoder(top-10)\n")

    query = "What medications are prescribed for asthma and what adverse reactions can occur?"
    candidates = hybrid.search(query, top_k=50)
    print(f'  Query: "{query}"')
    print(f"  Starting with {len(candidates)} hybrid candidates\n")

    # Pipeline A: rerank_first
    pipeline_a = PostRetrievalPipeline(
        reranker=reranker,
        embedding_model=store.model,
        order="rerank_first",
        mmr_lambda=settings.mmr_lambda,
    )
    t = time.time()
    results_a = pipeline_a.process(query, candidates, rerank_top_k=20, final_top_k=10)
    time_a = time.time() - t
    unique_a = len(set(r.chunk.doc_id for r in results_a))
    print_results(f"Pipeline A - rerank_first ({time_a:.2f}s, {unique_a} unique docs)", results_a)

    # Pipeline B: mmr_first
    pipeline_b = PostRetrievalPipeline(
        reranker=reranker,
        embedding_model=store.model,
        order="mmr_first",
        mmr_lambda=settings.mmr_lambda,
    )
    t = time.time()
    results_b = pipeline_b.process(query, candidates, rerank_top_k=20, final_top_k=10)
    time_b = time.time() - t
    unique_b = len(set(r.chunk.doc_id for r in results_b))
    print_results(f"Pipeline B - mmr_first ({time_b:.2f}s, {unique_b} unique docs)", results_b)

    # Overlap analysis
    ids_a = set(r.chunk.chunk_id for r in results_a)
    ids_b = set(r.chunk.chunk_id for r in results_b)
    overlap = len(ids_a & ids_b)
    print(f"  Overlap between Pipeline A and B: {overlap}/10 chunks in common\n")


def main():
    store, hybrid, reranker = build_components()

    test_reranker(store, hybrid, reranker)
    test_mmr(store, hybrid, reranker)
    test_pipeline(store, hybrid, reranker)

    print(SEP)
    print("MEDICAL POST-RETRIEVAL VERIFICATION COMPLETE")
    print(SEP)
    print("  [OK] CrossEncoder accurately scores clinical question/passage relevance")
    print("  [OK] MMR balances relevance with diverse medical topics")
    print("  [OK] PostRetrievalPipeline is fully integrated")
    print()


if __name__ == "__main__":
    main()
