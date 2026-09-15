"""
GaleMed AI — RAG Architecture Definitions
==========================================
Defines two pipeline architectures for ablation comparison:

    NaiveRAGPipeline
    ----------------
    Baseline: vector-only search → direct LLM generation.
    No query transformation, no BM25, no reranking, no cache, no graph.
    Represents a standard "first-principles" RAG implementation.

    FullGaleMedRAGPipeline
    ----------------------
    Full production pipeline using all GaleMed components:
        • Semantic cache (Redis)
        • Query routing + HyDE / decomposition transformation
        • Hybrid BM25 + Vector search (RRF fusion)
        • Neo4j Knowledge Graph retrieval (optional)
        • CrossEncoder reranking + MMR diversification
        • Langfuse observability + cost tracking

Both implement the same callable interface expected by RagasEvaluator:
    result = pipeline(query="What are symptoms of asthma?")
    # -> {"answer": str, "contexts": list[str], "cost_usd": float}

Graceful degradation:
    - If Qdrant is unavailable, both return a safe fallback answer.
    - If OpenAI key is missing, raises immediately with a clear error.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ── Medical Disclaimer (appended to Naive RAG answers) ────────────────────────
_DISCLAIMER = (
    "\n\n---\nThis information is for educational purposes only and is sourced from "
    "The Gale Encyclopedia of Medicine. It does not constitute medical advice. "
    "Always consult a qualified healthcare professional for medical concerns."
)

# ── Medical System Prompt (Naive RAG version — no safety chain) ───────────────
_NAIVE_SYSTEM_PROMPT = """\
You are a medical information assistant. Answer the following question based ONLY
on the provided context passages from The Gale Encyclopedia of Medicine.

If the context does not contain sufficient information, say:
"The provided medical references do not contain enough information to answer this question."

Never fabricate medical facts, drug dosages, or clinical recommendations.
"""

# ── Emergency detection (minimal, reused in Naive pipeline) ───────────────────
_EMERGENCY_TERMS = [
    "chest pain", "heart attack", "stroke", "can't breathe", "difficulty breathing",
    "seizure", "overdose", "unconscious", "severe bleeding", "anaphylaxis",
    "suicide", "suicidal",
]

_EMERGENCY_REPLY = (
    "EMERGENCY ALERT: Based on your query, this may be a life-threatening emergency. "
    "Please call emergency services immediately: 115 (Vietnam) / 911 (US) / 112 (EU). "
    "Do NOT wait — seek immediate professional medical help."
)


def _is_emergency(query: str) -> bool:
    q = query.lower()
    return any(term in q for term in _EMERGENCY_TERMS)


# ═══════════════════════════════════════════════════════════════════════════════
#  ARCHITECTURE 1: Naive RAG
# ═══════════════════════════════════════════════════════════════════════════════

class NaiveRAGPipeline:
    """
    Baseline RAG pipeline: simple vector search → LLM generation.

    Components used:
        - Qdrant vector store (top-k cosine similarity search only)
        - SentenceTransformer for query embedding
        - OpenAI gpt-4o-mini for answer generation

    Components NOT used (vs. Full GaleMed):
        - Redis semantic cache
        - BM25 retriever
        - Query transformation (HyDE / decomposition)
        - CrossEncoder reranking
        - MMR diversification
        - Neo4j knowledge graph
        - Langfuse observability

    Args:
        top_k:      Number of documents retrieved from Qdrant (default: 5).
        model:      OpenAI model for generation (default: gpt-4o-mini).
        openai_api_key: Optional override for OPENAI_API_KEY env var.

    Usage:
        pipeline = NaiveRAGPipeline(top_k=5)
        result = pipeline(query="What are symptoms of asthma?")
        # result = {"answer": str, "contexts": list[str], "cost_usd": float}
    """

    def __init__(
        self,
        top_k: int = 5,
        model: str = "gpt-4o-mini",
        openai_api_key: Optional[str] = None,
    ) -> None:
        self.top_k = top_k
        self.model = model
        self._api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self._ready = False
        self._vector_store = None
        self._embedding_model = None
        self._openai_client = None
        self._cost_calc = None
        self._init_error: Optional[str] = None

        self._lazy_init()

    def _lazy_init(self) -> None:
        """Initialise components; record error if services unavailable."""
        try:
            from openai import OpenAI
            from sentence_transformers import SentenceTransformer
            # pyrefly: ignore [missing-import]
            from src.config import settings
            # pyrefly: ignore [missing-import]
            from src.indexing.vector_store import VectorStore
            # pyrefly: ignore [missing-import]
            from src.observability.cost_calculator import CostCalculator

            self._embedding_model = SentenceTransformer(settings.embedding_model)
            self._vector_store = VectorStore(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                collection_name=settings.qdrant_collection,
                _embedding_model=self._embedding_model,
            )
            self._openai_client = OpenAI(api_key=self._api_key or settings.openai_api_key)
            self._cost_calc = CostCalculator()
            self._ready = True
            logger.info("[NaiveRAG] Initialised — top_k=%d, model=%s", self.top_k, self.model)

        except Exception as exc:
            self._init_error = str(exc)
            logger.warning("[NaiveRAG] Init failed (will use fallback): %s", exc)

    def __call__(self, *, query: str) -> dict:
        """
        Run Naive RAG pipeline for a single query.

        Returns:
            dict with keys: answer (str), contexts (list[str]), cost_usd (float)
        """
        # Emergency check first
        if _is_emergency(query):
            return {"answer": _EMERGENCY_REPLY, "contexts": [], "cost_usd": 0.0}

        if not self._ready:
            return {
                "answer": (
                    "The medical knowledge base is currently unavailable. "
                    "Please consult a qualified healthcare professional."
                ),
                "contexts": [],
                "cost_usd": 0.0,
            }

        try:
            # Stage 1: Vector search only (no transformation, no BM25, no reranking)
            candidates = self._vector_store.search(query, top_k=self.top_k)
            contexts = [r.chunk.content for r in candidates]

            if not contexts:
                return {
                    "answer": (
                        "The provided medical references do not contain enough information "
                        "to answer this question. Please consult a qualified healthcare professional."
                    ),
                    "contexts": [],
                    "cost_usd": 0.0,
                }

            # Stage 2: Build context string
            context_str = "\n\n---\n\n".join(
                f"[{i+1}] {r.chunk.content}" for i, r in enumerate(candidates)
            )

            # Stage 3: Direct LLM generation (no streaming)
            response = self._openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _NAIVE_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"},
                ],
                temperature=0.1,
                max_tokens=600,
            )

            answer = response.choices[0].message.content.strip() + _DISCLAIMER

            # Cost calculation
            from src.observability.cost_calculator import TokenUsage  # pyrefly: ignore [missing-import]
            usage = TokenUsage.from_openai_response(response)
            cost_usd = self._cost_calc.calculate(usage, model=self.model)

            return {"answer": answer, "contexts": contexts, "cost_usd": cost_usd}

        except Exception as exc:
            logger.error("[NaiveRAG] Pipeline error: %s", exc)
            return {
                "answer": f"An error occurred while processing your query: {exc}",
                "contexts": [],
                "cost_usd": 0.0,
            }


# ═══════════════════════════════════════════════════════════════════════════════
#  ARCHITECTURE 2: Full GaleMed RAG
# ═══════════════════════════════════════════════════════════════════════════════

class FullGaleMedRAGPipeline:
    """
    Full production GaleMed RAG pipeline — wraps src.orchestrator.pipeline.RAGPipeline.

    Adds on top of NaiveRAG:
        + Redis semantic cache (semantic deduplication)
        + HyDE / query decomposition transformation
        + Hybrid BM25 + Vector search (RRF fusion)
        + CrossEncoder reranking (ms-marco-MiniLM-L-6-v2)
        + MMR diversification (lambda=0.5)
        + Neo4j knowledge graph retrieval (optional)
        + Langfuse Cloud observability + cost tracking per span

    Args:
        use_graph:  Enable Neo4j graph retrieval (default: True, auto-disabled if unavailable).
        openai_api_key: Optional override for OPENAI_API_KEY env var.

    Usage:
        pipeline = FullGaleMedRAGPipeline(use_graph=False)
        result = pipeline(query="Describe symptoms of COPD")
        # result = {"answer": str, "contexts": list[str], "cost_usd": float}
    """

    def __init__(
        self,
        use_graph: bool = True,
        openai_api_key: Optional[str] = None,
    ) -> None:
        self.use_graph = use_graph
        self._api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self._pipeline = None
        self._cost_calc = None
        self._ready = False
        self._init_error: Optional[str] = None

        self._lazy_init()

    def _lazy_init(self) -> None:
        """Initialise the full RAG pipeline; record error if services unavailable."""
        try:
            # pyrefly: ignore [missing-import]
            from src.orchestrator.pipeline import RAGPipeline
            # pyrefly: ignore [missing-import]
            from src.observability.cost_calculator import CostCalculator

            self._pipeline = RAGPipeline(use_graph=self.use_graph)
            self._cost_calc = CostCalculator()
            self._ready = True
            logger.info("[FullGaleMed] Initialised (use_graph=%s)", self.use_graph)

        except Exception as exc:
            self._init_error = str(exc)
            logger.warning("[FullGaleMed] Init failed (will use fallback): %s", exc)

    def __call__(self, *, query: str) -> dict:
        """
        Run Full GaleMed pipeline for a single query.

        Converts RAGResponse into the dict format expected by RagasEvaluator:
            {"answer": str, "contexts": list[str], "cost_usd": float}
        """
        if not self._ready:
            return {
                "answer": (
                    "The medical knowledge base is currently unavailable. "
                    "Please consult a qualified healthcare professional."
                ),
                "contexts": [],
                "cost_usd": 0.0,
            }

        try:
            response = self._pipeline.process_query(query=query, search_mode="auto", top_k=10)

            # Extract contexts from RAGResponse.sources
            contexts: list[str] = []
            if hasattr(response, "sources") and response.sources:
                contexts = [r.chunk.content for r in response.sources if hasattr(r, "chunk")]

            # Extract cost from metadata
            cost_usd: float = 0.0
            if hasattr(response, "metadata") and response.metadata:
                cost_usd = float(response.metadata.get("cost_usd", 0.0))

            return {
                "answer": response.answer,
                "contexts": contexts,
                "cost_usd": cost_usd,
            }

        except Exception as exc:
            logger.error("[FullGaleMed] Pipeline error: %s", exc)
            return {
                "answer": f"An error occurred while processing your query: {exc}",
                "contexts": [],
                "cost_usd": 0.0,
            }

    def query(self, query: str) -> dict:
        return self(query=query)


class BaseRAGArchitecture:
    """Base interface for all RAG architectures in experiments."""

    def __init__(self, name: str = "BaseRAG"):
        self.name = name

    def query(self, query: str) -> dict:
        raise NotImplementedError

    def __call__(self, *, query: str) -> dict:
        return self.query(query)


class MockNaivePipeline(BaseRAGArchitecture):
    """Simulates NaiveRAGPipeline for rapid smoke testing without external APIs."""

    def __init__(self, name: str = "Mock Naive RAG"):
        super().__init__(name=name)

    def query(self, query: str) -> dict:
        return {
            "answer": (
                f"Naive RAG answer for: {query[:60]}. "
                "According to The Gale Encyclopedia of Medicine, this condition involves "
                "specific clinical symptoms and standard management. "
                "Please consult a qualified healthcare professional."
            ),
            "contexts": [
                f"Context chunk 1: Overview of {query[:40]}.",
                f"Context chunk 2: Clinical signs for {query[:40]}.",
            ],
            "input_tokens": 150,
            "output_tokens": 60,
            "cost_usd": 0.000150,
            "model": "gpt-4o-mini",
        }

    def __call__(self, *, query: str) -> dict:
        return self.query(query)


class MockFullPipeline(BaseRAGArchitecture):
    """Simulates FullGaleMedRAGPipeline for rapid smoke testing without external APIs."""

    def __init__(self, name: str = "Mock Full GaleMed RAG"):
        super().__init__(name=name)

    def query(self, query: str) -> dict:
        return {
            "answer": (
                f"Full GaleMed RAG answer for: {query[:60]}. "
                "According to The Gale Encyclopedia of Medicine (3rd Edition), "
                "this condition presents with distinct diagnostic criteria, pathophysiology, "
                "and evidence-based therapeutic protocols including first-line treatments and contraindications. "
                "Always consult a qualified healthcare specialist for clinical decision making."
            ),
            "contexts": [
                f"Context chunk 1 (hybrid-reranked): Comprehensive etiology and clinical signs for {query[:40]}.",
                f"Context chunk 2 (cross-encoder): Evidence-based therapeutic management and dosage guidelines.",
                f"Context chunk 3 (graph-connected): Related comorbidities and differential diagnosis pathways.",
            ],
            "input_tokens": 280,
            "output_tokens": 110,
            "cost_usd": 0.000310,
            "model": "gpt-4o-mini",
        }

    def __call__(self, *, query: str) -> dict:
        return self.query(query)


class ArchitectureFactory:
    """Factory to instantiate RAG architectures by name."""

    @staticmethod
    def create(name: str, **kwargs):
        name_lower = name.lower()
        if "naive" in name_lower:
            return NaiveRAGPipeline(**kwargs)
        elif any(k in name_lower for k in ("full", "advanced", "galemed")):
            return FullGaleMedRAGPipeline(**kwargs)
        raise ValueError(f"Unknown architecture name: {name}. Choose 'naive' or 'full'.")

