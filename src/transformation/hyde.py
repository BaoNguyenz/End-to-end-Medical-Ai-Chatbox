"""
hyde.py
Hypothetical Document Embeddings (HyDE) query transformation.

Instead of embedding the raw query, HyDE:
  1. Asks the LLM to write a hypothetical document that would answer the query.
  2. Embeds that hypothetical document.
  3. Uses the resulting embedding for vector search.

This bridges the vocabulary/style gap between short queries and long documents.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI
from sentence_transformers import SentenceTransformer

from src.models import SearchResult
from src.indexing.vector_store import VectorStore


_HYDE_PROMPT = """\
You are an expert technical writer at an enterprise software company.
Write a detailed, realistic paragraph (3-5 sentences) that would appear in
an official document and directly answers the following question.
Write in third-person, formal documentation style.
Do NOT start with "The answer is" or "Here is". Just write the paragraph.

Question: {query}

Hypothetical document excerpt:"""


class HyDE:
    """
    Hypothetical Document Embeddings transformation.

    Usage:
        hyde = HyDE(openai_client, model, embedding_model, cache_dir)
        results = hyde.search(query, vector_store, top_k=10)
    """

    def __init__(
        self,
        openai_client: OpenAI,
        model: str,
        embedding_model: SentenceTransformer,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.client = openai_client
        self.model = model
        self.embedding_model = embedding_model
        self.cache_dir = cache_dir or Path("cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self.cache_dir / "hyde_cache.json"
        self._cache: dict[str, str] = self._load_cache()

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _load_cache(self) -> dict[str, str]:
        if self._cache_file.exists():
            try:
                return json.loads(self._cache_file.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_cache(self) -> None:
        self._cache_file.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _cache_key(self, query: str) -> str:
        return hashlib.sha256(f"hyde:{self.model}:{query}".encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate_hypothetical_document(self, query: str) -> str:
        """
        Ask the LLM to write a hypothetical answer document for the query.
        Result is cached on disk to avoid repeated API calls.

        Returns:
            A hypothetical document string (not the query itself).
        """
        key = self._cache_key(query)
        if key in self._cache:
            print(f"[HyDE] Cache hit for query: '{query[:50]}...'")
            return self._cache[key]

        print(f"[HyDE] Generating hypothetical document for: '{query[:60]}'")
        t = time.time()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": _HYDE_PROMPT.format(query=query)},
            ],
            temperature=0.7,
            max_tokens=300,
        )

        hyp_doc = response.choices[0].message.content.strip()
        elapsed = time.time() - t
        print(f"[HyDE] Generated in {elapsed:.2f}s  ({len(hyp_doc)} chars)")

        self._cache[key] = hyp_doc
        self._save_cache()
        return hyp_doc

    # ------------------------------------------------------------------
    # Search interface
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        vector_store: VectorStore,
        top_k: int = 10,
        document_type: Optional[str] = None,
    ) -> tuple[list[SearchResult], str]:
        """
        HyDE search: generate hypothetical doc -> embed -> vector search.

        Returns:
            Tuple of (results, hypothetical_document).
        """
        hyp_doc = self.generate_hypothetical_document(query)

        # Embed the hypothetical document (not the original query)
        embedding = self.embedding_model.encode(
            [hyp_doc], normalize_embeddings=True, show_progress_bar=False
        )[0].tolist()

        results = vector_store.search_by_embedding(
            embedding=embedding,
            top_k=top_k,
            document_type=document_type,
        )

        return results, hyp_doc
