"""
post_retrieval_pipeline.py
Combines Cross-Encoder reranking and MMR diversification into a single pipeline.

Two supported orders:
  rerank_first (default):
    50 hybrid candidates -> CrossEncoder -> top 20 -> MMR -> top 10
    Best when: you want high-precision results, then diversify the top set.

  mmr_first:
    50 hybrid candidates -> MMR -> top 20 -> CrossEncoder -> top 10
    Best when: you want diverse candidates first, then pick the best from them.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from src.models import SearchResult
from src.post_retrieval.cross_encoder_reranker import CrossEncoderReranker
from src.post_retrieval.mmr import mmr_rerank


class PostRetrievalPipeline:
    """
    Two-stage post-retrieval processing: Reranker + MMR (or MMR + Reranker).

    Usage:
        pipeline = PostRetrievalPipeline(reranker, embedding_model, order="rerank_first")
        final_results = pipeline.process(query, candidates, rerank_top_k=20, final_top_k=10)
    """

    VALID_ORDERS = ("rerank_first", "mmr_first")

    def __init__(
        self,
        reranker: CrossEncoderReranker,
        embedding_model: SentenceTransformer,
        order: str = "rerank_first",
        mmr_lambda: float = 0.5,
    ) -> None:
        if order not in self.VALID_ORDERS:
            raise ValueError(f"order must be one of {self.VALID_ORDERS}, got '{order}'")

        self.reranker = reranker
        self.embedding_model = embedding_model
        self.order = order
        self.mmr_lambda = mmr_lambda

    def process(
        self,
        query: str,
        candidates: list[SearchResult],
        rerank_top_k: int = 20,
        final_top_k: int = 10,
        mmr_lambda: float | None = None,
    ) -> list[SearchResult]:
        """
        Run the two-stage post-retrieval pipeline.

        Args:
            query:         Original query string.
            candidates:    Initial retrieval results (typically top-50).
            rerank_top_k:  Intermediate size after first stage.
            final_top_k:   Final number of results to return.
            mmr_lambda:    Override instance mmr_lambda for this call.

        Returns:
            Final top-k SearchResults after both stages.
        """
        lam = mmr_lambda if mmr_lambda is not None else self.mmr_lambda

        if self.order == "rerank_first":
            # Stage 1: Cross-Encoder → keep top rerank_top_k
            after_rerank = self.reranker.rerank(
                query, candidates, top_k=rerank_top_k
            )
            # Stage 2: MMR diversification → keep final_top_k
            final = mmr_rerank(
                query, after_rerank, self.embedding_model,
                lambda_param=lam, top_k=final_top_k,
            )
            print(
                f"[Pipeline] rerank_first: {len(candidates)} -> "
                f"CrossEncoder({rerank_top_k}) -> MMR({final_top_k})"
            )

        else:  # mmr_first
            # Stage 1: MMR → diverse top rerank_top_k
            after_mmr = mmr_rerank(
                query, candidates, self.embedding_model,
                lambda_param=lam, top_k=rerank_top_k,
            )
            # Stage 2: Cross-Encoder → final precision ranking
            final = self.reranker.rerank(
                query, after_mmr, top_k=final_top_k
            )
            print(
                f"[Pipeline] mmr_first: {len(candidates)} -> "
                f"MMR({rerank_top_k}) -> CrossEncoder({final_top_k})"
            )

        return final
