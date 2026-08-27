"""
hybrid_search.py
Combines BM25 and Vector Search results using Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from src.models import Chunk, SearchResult, SearchSource
from src.retrieval.bm25_retriever import BM25Retriever
from src.indexing.vector_store import VectorStore


class HybridSearch:
    """
    Hybrid retrieval combining BM25 keyword search and vector semantic search
    using Reciprocal Rank Fusion (RRF) for score fusion.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_retriever: BM25Retriever,
    ) -> None:
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever

    # ------------------------------------------------------------------
    # RRF Fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _rrf_fuse(
        rankings: list[list[SearchResult]],
        weights: Optional[list[float]] = None,
        k: int = 60,
    ) -> list[SearchResult]:
        """
        Reciprocal Rank Fusion (RRF) of multiple ranked lists.

        Formula: RRF(d) = Σ weight_i / (k + rank_i(d))

        Args:
            rankings: List of ranked result lists.
            weights: Optional weight for each ranking list. Defaults to equal.
            k: RRF constant (default 60).

        Returns:
            Fused list of SearchResult sorted by RRF score descending.
        """
        if weights is None:
            weights = [1.0] * len(rankings)

        # Aggregate scores by chunk_id
        fused_scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}

        for ranking, weight in zip(rankings, weights):
            for rank, result in enumerate(ranking):
                cid = result.chunk.chunk_id
                rrf_contribution = weight / (k + rank + 1)  # rank is 0-based
                fused_scores[cid] = fused_scores.get(cid, 0.0) + rrf_contribution
                if cid not in chunk_map:
                    chunk_map[cid] = result.chunk

        # Sort by fused score
        sorted_items = sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            SearchResult(
                chunk=chunk_map[cid],
                score=score,
                source=SearchSource.HYBRID,
            )
            for cid, score in sorted_items
        ]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 20,
        bm25_top_k: int = 50,
        vector_top_k: int = 50,
        bm25_weight: float = 1.0,
        vector_weight: float = 1.0,
        rrf_k: int = 60,
    ) -> list[SearchResult]:
        """
        Execute hybrid search combining BM25 and Vector results with RRF.

        Args:
            query: The search query.
            top_k: Number of final results after fusion.
            bm25_top_k: Number of BM25 candidates.
            vector_top_k: Number of vector candidates.
            bm25_weight: Weight for BM25 rankings in RRF.
            vector_weight: Weight for vector rankings in RRF.
            rrf_k: RRF constant k.

        Returns:
            Fused search results.
        """
        # Run BM25 and Vector search in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            bm25_future = executor.submit(
                self.bm25_retriever.search, query, bm25_top_k
            )
            vector_future = executor.submit(
                self.vector_store.search, query, vector_top_k
            )
            bm25_results = bm25_future.result()
            vector_results = vector_future.result()

        # Fuse with RRF
        fused = self._rrf_fuse(
            rankings=[bm25_results, vector_results],
            weights=[bm25_weight, vector_weight],
            k=rrf_k,
        )

        return fused[:top_k]

    def search_bm25_only(self, query: str, top_k: int = 20) -> list[SearchResult]:
        """Convenience: BM25 search only."""
        return self.bm25_retriever.search(query, top_k)

    def search_vector_only(self, query: str, top_k: int = 20) -> list[SearchResult]:
        """Convenience: Vector search only."""
        return self.vector_store.search(query, top_k)
