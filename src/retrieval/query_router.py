"""
query_router.py
Medical AI Query Router - Classifies incoming medical queries and
routes them to the optimal search strategy.

Query types recognized:
- KEYWORD  → Drug names, ICD codes, dosages → BM25-weighted search
- SEMANTIC → Symptom descriptions, conceptual questions → Vector-weighted search
- GRAPH    → Relationship questions (interactions, contraindications) → GraphRAG
- HYBRID   → General medical questions → balanced fusion (default)
"""

from __future__ import annotations

import re

from src.models import SearchResult, SearchStrategy, QueryType
from src.retrieval.hybrid_search import HybridSearch


# ── Medical Keyword Patterns (BM25-heavy) ─────────────────────────────────────
_KEYWORD_PATTERNS = [
    # Drug dosages: "500mg", "10 ml", "20mcg", "2.5mg/kg"
    r"\d+\s*(mg|ml|mcg|g|iu|mEq|mmol)(\s*/\s*\w+)?",
    # ICD-10 codes: "J45", "E11.9", "I10"
    r"\b[A-Z]\d{2}(\.\d{1,2})?\b",
    # Drug names with brand/generic suffix pattern: "Albuterol (Ventolin)"
    r"\b\w+\s*\(\w+\)\b",
    # Lab test reference codes: "CBC", "CRP", "LDL", "HbA1c"
    r"\b(CBC|CRP|LDL|HDL|HbA1c|ESR|TSH|PSA|INR|AST|ALT|eGFR)\b",
    # Medical procedure codes: "CPT", "ICD" references
    r"\bCPT[-\s]?\d{4,5}\b",
    # Version / edition references
    r"\d+\.\d+\.\d+",
]

# ── Medical Semantic Patterns (Vector-heavy) ──────────────────────────────────
_SEMANTIC_PATTERNS = [
    # Symptom-based queries
    r"\b(symptom|sign|feel|feeling|hurt|pain|ache|suffer|complain|experience|wheez|cough|fever|rash|swell|bleed|nausea|vomit|dizzin|fatigue|headache)\b",
    # Causation/mechanism questions
    r"^(explain|describe|what is|what are|how does|why|how do|what causes)\b",
    # Comparison/overview
    r"\b(difference between|compare|similar to|overview|relationship between|impact of)\b",
    # Treatment/management questions
    r"\b(treatment|manage|therapy|cure|remedy|relief|prevent|prognosis)\b",
    # Clinical description
    r"\b(diagnosis|diagnose|test for|screening|detect|risk|procedure|surgery|prognosis|complication)\b",
]

# ── Medical Graph/Relationship Patterns (GraphRAG) ───────────────────────────
_GRAPH_PATTERNS = [
    # Drug interactions
    r"\b(interaction|interact|combined with|take together|drug.drug)\b",
    # Contraindications
    r"\b(contraindic|should not take|avoid if|not safe for|forbidden|unsafe)\b",
    # Side effects / adverse events
    r"\b(side effect|adverse|complication|reaction|risk of)\b",
    # Relationships between entities
    r"\b(what drug|which medication|what treat|associated with|linked to|cause of|lead to)\b",
]


class QueryRouter:
    """
    Medical AI Query Router.

    Analyzes query intent and routes to the best search strategy:
    - KEYWORD  → BM25-weighted search (drug names, dosages, lab codes, ICD codes)
    - SEMANTIC → Vector-weighted search (symptoms, conceptual explanations)
    - GRAPH    → GraphRAG on Neo4j (drug interactions, contraindications, relationships)
    - HYBRID   → Balanced RRF fusion (default for general medical questions)
    """

    def classify(self, query: str) -> tuple[QueryType, SearchStrategy]:
        """
        Classify medical query type and determine search strategy.

        Returns:
            Tuple of (QueryType, SearchStrategy).
        """
        query_lower = query.lower().strip()

        # Priority 1: Graph/relationship queries → GraphRAG
        for pattern in _GRAPH_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return QueryType.COMPLEX, SearchStrategy.HYBRID  # Graph handled in pipeline

        # Priority 2: Keyword-heavy (drug names, codes, dosages)
        for pattern in _KEYWORD_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.KEYWORD, SearchStrategy.BM25

        # Priority 3: Semantic (symptom descriptions, explanations)
        for pattern in _SEMANTIC_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return QueryType.SEMANTIC, SearchStrategy.VECTOR

        # Default: Hybrid balanced search
        return QueryType.SIMPLE, SearchStrategy.HYBRID

    def route(
        self,
        query: str,
        hybrid_search: HybridSearch,
        top_k: int = 20,
        rrf_k: int = 60,
    ) -> tuple[list[SearchResult], QueryType, SearchStrategy]:
        """
        Classify query and execute the appropriate medical search strategy.

        Args:
            query:        User medical query.
            hybrid_search: HybridSearch instance.
            top_k:        Number of results to return.
            rrf_k:        RRF constant.

        Returns:
            Tuple of (results, query_type, search_strategy).
        """
        query_type, strategy = self.classify(query)

        if strategy == SearchStrategy.BM25:
            # Drug name / code queries: weight BM25 heavily for exact term matching
            results = hybrid_search.search(
                query, top_k=top_k,
                bm25_weight=3.0, vector_weight=1.0,
                rrf_k=rrf_k,
            )
        elif strategy == SearchStrategy.VECTOR:
            # Symptom / conceptual queries: weight vector for semantic understanding
            results = hybrid_search.search(
                query, top_k=top_k,
                bm25_weight=1.0, vector_weight=3.0,
                rrf_k=rrf_k,
            )
        else:
            # Hybrid: balanced for general medical questions
            results = hybrid_search.search(
                query, top_k=top_k,
                bm25_weight=1.5, vector_weight=2.0,
                rrf_k=rrf_k,
            )

        return results, query_type, strategy
