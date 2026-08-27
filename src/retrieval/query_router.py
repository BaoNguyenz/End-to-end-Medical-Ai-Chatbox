"""
query_router.py
Classifies incoming queries and routes them to the optimal search strategy.
"""

from __future__ import annotations

import re

from src.models import SearchResult, SearchStrategy, QueryType
from src.retrieval.hybrid_search import HybridSearch


# Patterns that indicate keyword-heavy queries
_KEYWORD_PATTERNS = [
    r"ERR_\w+",             # Error codes
    r"[A-Z]{2,}-\d+",       # Product/policy IDs like TDPRO-2024, POL-001
    r"\d+\.\d+\.\d+",       # Version numbers like 2.3.1
    r"(GET|POST|PUT|DELETE|PATCH)\s+/",  # HTTP methods
    r"REG-\w+-\d+",         # Register codes
]

# Patterns that indicate semantic/conceptual queries
_SEMANTIC_PATTERNS = [
    r"^(explain|describe|how does|what is|why|what are)\b",
    r"\b(difference|compare|overview|relationship|impact)\b",
    r"\b(benefits?|advantages?|disadvantages?|purpose)\b",
]


class QueryRouter:
    """
    Analyzes query intent and routes to the best search strategy:
    - KEYWORD: BM25-weighted search (error codes, IDs, version numbers)
    - SEMANTIC: Vector-weighted search (conceptual questions)
    - HYBRID: Balanced RRF fusion (default)
    """

    def classify(self, query: str) -> tuple[QueryType, SearchStrategy]:
        """
        Classify query type and determine search strategy.

        Returns:
            Tuple of (QueryType, SearchStrategy).
        """
        # Check for keyword patterns
        for pattern in _KEYWORD_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.KEYWORD, SearchStrategy.BM25

        # Check for semantic patterns
        query_lower = query.lower().strip()
        for pattern in _SEMANTIC_PATTERNS:
            if re.search(pattern, query_lower):
                return QueryType.SEMANTIC, SearchStrategy.VECTOR

        # Default to hybrid
        return QueryType.SIMPLE, SearchStrategy.HYBRID

    def route(
        self,
        query: str,
        hybrid_search: HybridSearch,
        top_k: int = 20,
        rrf_k: int = 60,
    ) -> tuple[list[SearchResult], QueryType, SearchStrategy]:
        """
        Classify query and execute the appropriate search strategy.

        Args:
            query: User query.
            hybrid_search: HybridSearch instance.
            top_k: Number of results to return.
            rrf_k: RRF constant.

        Returns:
            Tuple of (results, query_type, search_strategy).
        """
        query_type, strategy = self.classify(query)

        if strategy == SearchStrategy.BM25:
            # Keyword-heavy: weight BM25 much higher
            results = hybrid_search.search(
                query, top_k=top_k,
                bm25_weight=3.0, vector_weight=1.0,
                rrf_k=rrf_k,
            )
        elif strategy == SearchStrategy.VECTOR:
            # Semantic: weight vector higher
            results = hybrid_search.search(
                query, top_k=top_k,
                bm25_weight=1.0, vector_weight=3.0,
                rrf_k=rrf_k,
            )
        else:
            # Hybrid: equal weights
            results = hybrid_search.search(
                query, top_k=top_k,
                bm25_weight=1.0, vector_weight=1.0,
                rrf_k=rrf_k,
            )

        return results, query_type, strategy
