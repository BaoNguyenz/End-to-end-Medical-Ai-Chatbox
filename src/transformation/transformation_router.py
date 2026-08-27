"""
transformation_router.py
Decides which query transformation to apply (or none) before retrieval.

Decision logic:
  - complex  -> QueryDecomposer  (multi-part, "compare X and Y", conjunctions)
  - vague    -> HyDE             (short/ambiguous, needs hypothetical grounding)
  - simple   -> direct search    (no transformation needed)
"""

from __future__ import annotations

import re
from typing import Optional

from openai import OpenAI
from sentence_transformers import SentenceTransformer

from src.models import SearchResult
from src.indexing.vector_store import VectorStore
from src.retrieval.hybrid_search import HybridSearch
from src.transformation.hyde import HyDE
from src.transformation.query_decomposition import QueryDecomposer


# Query is "vague" if it matches these patterns
_VAGUE_PATTERNS = [
    r"^(tell me about|what about|info on|information about|explain)\s+\w+(\s+\w+)?$",
    r"^(overview|summary|details|more about)\s+",
]


class TransformationRouter:
    """
    Routes incoming queries to the appropriate transformation:

    'complex'  -> decompose into sub-questions, search each, aggregate
    'vague'    -> HyDE: generate hypothetical doc, embed, search
    'simple'   -> direct hybrid search (no LLM needed)

    Usage:
        router = TransformationRouter(openai_client, model, embedding_model, cache_dir)
        results, meta = router.transform_and_search(query, vector_store, hybrid_search, top_k=10)
    """

    def __init__(
        self,
        openai_client: OpenAI,
        model: str,
        embedding_model: SentenceTransformer,
        cache_dir=None,
    ) -> None:
        from pathlib import Path
        self.hyde = HyDE(
            openai_client=openai_client,
            model=model,
            embedding_model=embedding_model,
            cache_dir=cache_dir,
        )
        self.decomposer = QueryDecomposer(
            openai_client=openai_client,
            model=model,
            cache_dir=cache_dir,
        )

    # ------------------------------------------------------------------
    # Classification (no LLM - pure heuristics)
    # ------------------------------------------------------------------

    def classify(self, query: str) -> str:
        """
        Classify query into one of: 'complex', 'vague', 'simple'.

        Returns:
            str: One of 'complex', 'vague', 'simple'.
        """
        # Complex: multi-clause, conjunctions, compare/contrast
        if self.decomposer.is_complex(query):
            return "complex"

        # Vague: very short OR matches vague patterns
        word_count = len(query.split())
        if word_count <= 3:
            return "vague"

        q_lower = query.lower().strip()
        for pattern in _VAGUE_PATTERNS:
            if re.match(pattern, q_lower):
                return "vague"

        return "simple"

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def transform_and_search(
        self,
        query: str,
        vector_store: VectorStore,
        hybrid_search: HybridSearch,
        top_k: int = 10,
        document_type: Optional[str] = None,
    ) -> tuple[list[SearchResult], dict]:
        """
        Classify the query, apply the right transformation, and execute search.

        Returns:
            Tuple of:
              - results: list[SearchResult]
              - meta: dict with keys 'query_class', 'transformation', 'extra'
                (extra: hypothetical_doc for HyDE, sub_queries for decompose)
        """
        query_class = self.classify(query)
        meta: dict = {"query_class": query_class, "transformation": query_class, "extra": None}

        if query_class == "complex":
            print(f"[TransformRouter] Complex query -> decomposing")
            results, sub_queries = self.decomposer.search(query, hybrid_search, top_k=top_k)
            meta["extra"] = sub_queries

        elif query_class == "vague":
            print(f"[TransformRouter] Vague query -> HyDE")
            results, hyp_doc = self.hyde.search(
                query, vector_store, top_k=top_k, document_type=document_type
            )
            meta["extra"] = hyp_doc

        else:
            print(f"[TransformRouter] Simple query -> direct hybrid search")
            results = hybrid_search.search(query, top_k=top_k)
            meta["transformation"] = "none"

        return results, meta
