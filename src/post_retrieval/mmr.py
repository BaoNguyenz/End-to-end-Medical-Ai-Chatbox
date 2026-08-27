"""
mmr.py
Maximal Marginal Relevance (MMR) for result diversification.

Problem MMR solves:
  Pure relevance ranking often returns many near-duplicate chunks from the same section.
  MMR balances relevance (similarity to query) with diversity (dissimilarity to already-selected chunks).

MMR formula (Carbonell & Goldstein, 1998):
  MMR = argmax [ λ * sim(d, query) - (1-λ) * max_sim(d, selected) ]

  λ = 1.0 → pure relevance (no diversity)
  λ = 0.0 → pure diversity (no relevance)
  λ = 0.5 → balanced (recommended default)
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from src.models import SearchResult


def mmr_rerank(
    query: str,
    results: list[SearchResult],
    embedding_model: SentenceTransformer,
    lambda_param: float = 0.5,
    top_k: int = 10,
) -> list[SearchResult]:
    """
    Re-rank results using Maximal Marginal Relevance (MMR).

    Args:
        query:           Original query text (used to compute query embedding).
        results:         Candidate SearchResults (pre-retrieved and optionally pre-reranked).
        embedding_model: SentenceTransformer for computing embeddings.
        lambda_param:    λ in [0, 1]. Higher = more relevance, lower = more diversity.
        top_k:           Number of results to select.

    Returns:
        Top-k SearchResults selected by MMR (order matters — greedy selection).
    """
    if not results:
        return []

    top_k = min(top_k, len(results))

    # Encode query and all candidate passages
    texts = [query] + [r.chunk.content for r in results]
    embeddings = embedding_model.encode(
        texts, normalize_embeddings=True, show_progress_bar=False
    )

    query_emb = embeddings[0]           # shape: (dim,)
    doc_embs = embeddings[1:]           # shape: (n_candidates, dim)

    # Cosine similarity to query for each doc (embeddings already normalized)
    query_sims = doc_embs @ query_emb   # shape: (n_candidates,)

    selected_indices: list[int] = []
    remaining_indices = list(range(len(results)))

    while len(selected_indices) < top_k and remaining_indices:
        if not selected_indices:
            # First selection: pick the most relevant doc
            best_idx = max(remaining_indices, key=lambda i: query_sims[i])
        else:
            # MMR selection
            selected_embs = doc_embs[selected_indices]  # shape: (k, dim)

            def mmr_score(i: int) -> float:
                relevance = query_sims[i]
                # Max cosine similarity to any already-selected doc
                redundancy = float(np.max(doc_embs[i] @ selected_embs.T))
                return lambda_param * relevance - (1 - lambda_param) * redundancy

            best_idx = max(remaining_indices, key=mmr_score)

        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    # Build output with MMR rank-based scores (1.0, 0.9, 0.8, ...)
    # We replace scores with descending values to indicate MMR rank order
    return [
        SearchResult(
            chunk=results[i].chunk,
            score=1.0 - (rank * 0.1),   # rank-based score: 1.0, 0.9, 0.8, ...
            source=results[i].source,
        )
        for rank, i in enumerate(selected_indices)
    ]
