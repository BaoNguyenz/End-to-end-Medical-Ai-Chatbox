"""
cross_encoder_reranker.py
Re-ranks retrieval results using a Cross-Encoder model.

Why Cross-Encoder?
  - Bi-encoder (SentenceTransformer) encodes query & doc SEPARATELY → fast but less accurate.
  - Cross-Encoder takes (query, doc) as a PAIR → captures fine-grained interaction → much more accurate.
  - Typical pipeline: bi-encoder retrieves top-50 candidates, cross-encoder re-ranks to top-10.
"""

from __future__ import annotations

import time

from sentence_transformers import CrossEncoder

from src.models import SearchResult


class CrossEncoderReranker:
    """
    Re-ranks a list of SearchResults using a Cross-Encoder model.

    The cross-encoder scores each (query, chunk) pair and re-orders results
    by relevance, often significantly improving precision@k.

    Model default: cross-encoder/ms-marco-MiniLM-L-6-v2
      - Trained on MS MARCO passage ranking
      - ~22M params, runs well on CPU
      - Returns raw logit scores (higher = more relevant)
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        print(f"[CrossEncoder] Loading model: {model_name}")
        t = time.time()
        # device=None → auto-detect GPU/CPU
        self.model = CrossEncoder(model_name, max_length=512)
        print(f"[CrossEncoder] Model loaded in {time.time()-t:.2f}s")
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 10,
        batch_size: int = 32,
    ) -> list[SearchResult]:
        """
        Re-rank results using cross-encoder scores.

        Args:
            query:      The original search query.
            results:    Candidate SearchResults (typically top-50 from hybrid search).
            top_k:      Number of results to return after reranking.
            batch_size: Batch size for model inference (tune for your hardware).

        Returns:
            Top-k SearchResults sorted by cross-encoder score (highest first).
            The .score field is replaced with the cross-encoder logit score.
        """
        if not results:
            return []

        # Build (query, passage) pairs
        pairs = [(query, r.chunk.content) for r in results]

        # Batch predict scores
        t = time.time()
        scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        elapsed = time.time() - t
        print(
            f"[CrossEncoder] Scored {len(results)} candidates in {elapsed:.2f}s "
            f"-> keeping top {min(top_k, len(results))}"
        )

        # Attach scores and sort
        scored = sorted(
            zip(scores, results),
            key=lambda x: x[0],
            reverse=True,
        )

        # Return new SearchResult objects with cross-encoder scores
        reranked = []
        for score, result in scored[:top_k]:
            reranked.append(SearchResult(
                chunk=result.chunk,
                score=float(score),
                source=result.source,
            ))

        return reranked
