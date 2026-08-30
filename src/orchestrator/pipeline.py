"""
pipeline.py
Unified Medical AI RAG pipeline orchestrating all Task 1-5 components.

Query flow:
  1. QueryRouter      → classify query type & search strategy
  2. TransformRouter  → HyDE / Decompose / Direct based on query class
  3. Hybrid Search    → BM25 + Vector (RRF fused)
  4. Graph Search     → Neo4j NL-to-Cypher (optional)
  5. Merge results    → combine vector + graph candidates
  6. PostRetrieval    → CrossEncoder rerank + MMR diversify
  7. LLM Generation   → medical answer with disclaimer

Safety features:
  - Emergency detection: recognizes life-threatening symptom queries
  - Medical disclaimer: always appended to AI answers
  - Grounding: ONLY answers from Gale Encyclopedia context
"""

from __future__ import annotations

import re
import time
from typing import Optional

from openai import OpenAI
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.models import SearchResult, RAGResponse
from src.indexing.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.query_router import QueryRouter
from src.transformation.transformation_router import TransformationRouter
from src.post_retrieval.cross_encoder_reranker import CrossEncoderReranker
from src.post_retrieval.post_retrieval_pipeline import PostRetrievalPipeline


# ── Medical System Prompt ─────────────────────────────────────────────────────
_ANSWER_SYSTEM = """\
You are a clinical medical information assistant powered by The Gale Encyclopedia of Medicine (3rd Edition).

Your role is to provide accurate, evidence-based medical information based ONLY on the provided context.
Follow these rules strictly:
1. Answer using ONLY information from the provided context passages.
2. Cite the source document when referencing specific facts (e.g., "According to the Asthma entry...").
3. Use clear, accessible language — explain medical terms when they first appear.
4. Organize your answer with clear structure when answering multi-part questions.
5. If the context does NOT contain sufficient information to answer, say exactly:
   "The provided medical references do not contain enough information to fully answer this question.
    Please consult a qualified healthcare professional."
6. NEVER fabricate drug dosages, diagnostic criteria, or clinical recommendations.
7. NEVER provide a personal diagnosis or recommend specific treatment for the user's condition.

After every answer, ALWAYS append this disclaimer on a new line:
---
⚠️ *This information is for educational purposes only and is sourced from The Gale Encyclopedia of Medicine.
It does not constitute medical advice. Always consult a qualified healthcare professional for medical concerns.*
"""

# ── Emergency Keywords ────────────────────────────────────────────────────────
_EMERGENCY_PATTERNS = [
    r"\b(suicide|suicidal|kill (my)?self|end (my )?life|want to die)\b",
    r"\b(chest pain|heart attack|stroke|can't breathe|difficulty breathing|seizure)\b",
    r"\b(overdose|poisoning|unconscious|unresponsive|severe allergic reaction|anaphylaxis)\b",
    r"\b(severe bleeding|hemorrhage|choking|drowning|electric shock)\b",
]

_EMERGENCY_RESPONSE = """\
🚨 **EMERGENCY ALERT**

Based on your query, this may be a medical emergency.

**Please take immediate action:**
- **Call emergency services immediately: 113 (Vietnam) / 911 (US) / 999 (UK) / 112 (EU)**
- Go to the nearest emergency room
- Do NOT wait — seek immediate professional medical help

The AI assistant cannot provide emergency medical guidance. Your safety is the priority.
"""


class RAGPipeline:
    """
    Unified Medical AI RAG pipeline integrating all components.

    Can be initialized with graph support disabled if Neo4j is unavailable.
    """

    def __init__(
        self,
        use_graph: bool = True,
        pipeline_order: str = "rerank_first",
    ) -> None:
        self.latency: dict[str, float] = {}
        self.use_graph = use_graph

        # ── Core models (shared across components) ──────────────────────────
        t = time.time()
        self._embedding_model = SentenceTransformer(settings.embedding_model)
        self.latency["init_embedding"] = time.time() - t

        # ── Task 1: Vector Store ────────────────────────────────────────────
        t = time.time()
        self.vector_store = VectorStore(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=settings.qdrant_collection,
            _embedding_model=self._embedding_model,
        )

        # ── Task 2: Hybrid Search ───────────────────────────────────────────
        t = time.time()
        all_chunks = self.vector_store.get_all_chunks()
        self.bm25 = BM25Retriever()
        self.bm25.build_index(all_chunks)
        self.latency["init_bm25"] = time.time() - t

        self.hybrid_search = HybridSearch(self.vector_store, self.bm25)
        self.query_router = QueryRouter()

        # ── Task 3: Transformation Router ───────────────────────────────────
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        self.transform_router = TransformationRouter(
            openai_client=self.openai_client,
            model=settings.openai_model,
            embedding_model=self._embedding_model,
            cache_dir=settings.cache_dir,
        )

        # ── Task 4: Post-Retrieval ──────────────────────────────────────────
        t = time.time()
        self.reranker = CrossEncoderReranker(settings.cross_encoder_model)
        self.latency["init_reranker"] = time.time() - t

        self.post_pipeline = PostRetrievalPipeline(
            reranker=self.reranker,
            embedding_model=self._embedding_model,
            order=pipeline_order,
            mmr_lambda=settings.mmr_lambda,
        )

        # ── Task 5: Graph (optional) ────────────────────────────────────────
        self.graph_retriever = None
        if use_graph:
            try:
                from src.graph.knowledge_graph import KnowledgeGraph
                from src.graph.graph_retriever import GraphRetriever
                kg = KnowledgeGraph(
                    uri=settings.neo4j_uri,
                    user=settings.neo4j_user,
                    password=settings.neo4j_password,
                )
                self.graph_retriever = GraphRetriever(kg, self.openai_client, settings.openai_model)
                print("[RAGPipeline] Medical Knowledge Graph retriever enabled")
            except Exception as e:
                print(f"[RAGPipeline] Graph disabled (Neo4j unavailable): {e}")

        print("[RAGPipeline] Medical AI pipeline initialized.")

    # ------------------------------------------------------------------
    # Emergency detection
    # ------------------------------------------------------------------

    def _is_emergency(self, query: str) -> bool:
        """Detect potential medical emergency in user query."""
        query_lower = query.lower()
        for pattern in _EMERGENCY_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return True
        return False

    # ------------------------------------------------------------------
    # Main query method
    # ------------------------------------------------------------------

    def process_query(
        self,
        query: str,
        search_mode: str = "auto",   # "auto" | "hybrid" | "vector" | "bm25" | "graph"
        top_k: int = 10,
        use_graph: Optional[bool] = None,
        rerank_top_k: int = 20,
    ) -> RAGResponse:
        """
        Process a medical query through the full RAG pipeline.

        Args:
            query:        The user's medical question.
            search_mode:  "auto" uses QueryRouter; others force a strategy.
            top_k:        Final number of context chunks for answer generation.
            use_graph:    Override instance-level graph setting for this query.
            rerank_top_k: Candidates passed to CrossEncoder before MMR.

        Returns:
            RAGResponse with medical answer, sources, latency breakdown, and metadata.
        """
        total_start = time.time()
        latency: dict[str, float] = {}
        metadata: dict = {"search_mode": search_mode, "query_class": "unknown"}

        # ── Emergency check ────────────────────────────────────────────────
        if self._is_emergency(query):
            return RAGResponse(
                query=query,
                answer=_EMERGENCY_RESPONSE,
                sources=[],
                latency={"total": time.time() - total_start},
                metadata={"emergency": True, "search_mode": "none"},
            )

        # ── Stage 1: Query classification ──────────────────────────────────
        t = time.time()
        _, strategy = self.query_router.classify(query)
        latency["query_classify"] = time.time() - t
        metadata["strategy"] = strategy.value

        # ── Stage 2: Query transformation + retrieval ─────────────────────
        t = time.time()
        if search_mode == "auto":
            candidates, transform_meta = self.transform_router.transform_and_search(
                query, self.vector_store, self.hybrid_search, top_k=50
            )
            metadata["query_class"] = transform_meta["query_class"]
            metadata["transformation"] = transform_meta["transformation"]
        elif search_mode == "vector":
            candidates = self.vector_store.search(query, top_k=50)
            metadata["query_class"] = "simple"
            metadata["transformation"] = "none"
        elif search_mode == "bm25":
            candidates = self.bm25.search(query, top_k=50)
            metadata["query_class"] = "keyword"
            metadata["transformation"] = "none"
        else:  # "hybrid" or fallback
            candidates = self.hybrid_search.search(query, top_k=50)
            metadata["query_class"] = "simple"
            metadata["transformation"] = "none"
        latency["retrieval"] = time.time() - t

        # ── Stage 3: Graph retrieval (merge) ──────────────────────────────
        graph_results: list[SearchResult] = []
        _use_graph = use_graph if use_graph is not None else (self.use_graph and self.graph_retriever is not None)
        if _use_graph and self.graph_retriever and search_mode in ("auto", "graph"):
            t = time.time()
            try:
                graph_results = self.graph_retriever.search(query)
            except Exception as e:
                print(f"[RAGPipeline] Medical graph search failed: {e}")
            latency["graph_search"] = time.time() - t

        # Merge: vector candidates first, then graph results appended
        all_candidates = candidates + graph_results

        # ── Stage 4: Post-retrieval (rerank + MMR) ────────────────────────
        t = time.time()
        final_results = self.post_pipeline.process(
            query, all_candidates,
            rerank_top_k=rerank_top_k,
            final_top_k=top_k,
        )
        latency["post_retrieval"] = time.time() - t

        # ── Stage 5: Answer generation ────────────────────────────────────
        t = time.time()
        context_parts = []
        for i, r in enumerate(final_results):
            # Include the encyclopedia entry title in the source label
            source_label = r.chunk.doc_id.replace("_", " ").title()
            context_parts.append(
                f"[{i+1}] Source: {source_label} (doc_id: {r.chunk.doc_id})\n{r.chunk.content}"
            )
        context = "\n\n---\n\n".join(context_parts)

        llm_response = self.openai_client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": _ANSWER_SYSTEM},
                {"role": "user", "content": f"Medical Reference Context:\n{context}\n\nPatient Question: {query}"},
            ],
            temperature=0.1,   # Very low temperature for factual medical answers
            max_tokens=800,
        )
        answer = llm_response.choices[0].message.content.strip()
        latency["generation"] = time.time() - t

        latency["total"] = time.time() - total_start
        metadata["num_candidates"] = len(candidates)
        metadata["num_graph_results"] = len(graph_results)
        metadata["num_final"] = len(final_results)

        return RAGResponse(
            query=query,
            answer=answer,
            sources=final_results,
            latency=latency,
            metadata=metadata,
        )
