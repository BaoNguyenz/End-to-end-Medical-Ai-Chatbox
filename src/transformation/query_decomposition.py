"""
query_decomposition.py
Breaks complex multi-part queries into simpler independent sub-queries.

Example:
  Input:  "Compare data privacy policy and remote work policy, list stakeholders for both"
  Output: ["What does the data privacy policy cover?",
           "What does the remote work policy cover?",
           "Who are the stakeholders for the data privacy policy?",
           "Who are the stakeholders for the remote work policy?"]
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI

from src.models import SearchResult
from src.retrieval.hybrid_search import HybridSearch


_DECOMPOSE_PROMPT = """\
You are a query analysis expert. Your task is to break down a complex question
into 2-4 simpler, independent sub-questions that together cover all aspects
of the original question.

Rules:
- Each sub-question must be self-contained (answerable without the others).
- Keep sub-questions focused and specific.
- Return ONLY a JSON array of strings. No explanations, no markdown.

Example input: "Compare data privacy policy and remote work policy and list their stakeholders"
Example output: ["What are the main rules in the data privacy policy?", "What are the main rules in the remote work policy?", "Who are the stakeholders for data privacy?", "Who are the stakeholders for remote work?"]

Now decompose this question:
{query}

JSON array:"""


# Patterns that indicate a complex multi-part query
_COMPLEX_PATTERNS = [
    r"\b(and|also|as well as|furthermore|additionally)\b",
    r"\b(compare|contrast|difference between|versus|vs\.?)\b",
    r"\b(both|all of|each of|every)\b",
    r"\b(list|enumerate|describe each)\b.{5,}(and|also)\b",
    r",\s*and\b",          # "X, and Y" structure
    r"\?.*\?",             # Multiple question marks
]


class QueryDecomposer:
    """
    Decomposes complex queries into independent sub-queries using an LLM.

    Usage:
        decomposer = QueryDecomposer(openai_client, model, cache_dir)
        sub_queries = decomposer.decompose(query)
        results = decomposer.search(query, hybrid_search, top_k=20)
    """

    def __init__(
        self,
        openai_client: OpenAI,
        model: str,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.client = openai_client
        self.model = model
        self.cache_dir = cache_dir or Path("cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self.cache_dir / "decompose_cache.json"
        self._cache: dict[str, list[str]] = self._load_cache()

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _load_cache(self) -> dict[str, list[str]]:
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
        return hashlib.sha256(f"decompose:{self.model}:{query}".encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Complexity detection (no LLM needed)
    # ------------------------------------------------------------------

    def is_complex(self, query: str) -> bool:
        """
        Heuristic check: does this query contain multiple sub-questions?
        Returns True if the query likely needs decomposition.
        """
        q = query.lower()
        for pattern in _COMPLEX_PATTERNS:
            if re.search(pattern, q, re.IGNORECASE):
                return True
        # Also complex if query is very long (likely multi-clause)
        return len(query.split()) > 20

    # ------------------------------------------------------------------
    # Core decomposition
    # ------------------------------------------------------------------

    def decompose(self, query: str) -> list[str]:
        """
        Ask the LLM to decompose the query into sub-questions.
        Returns the original query as a single-item list if decomposition fails.

        Cached on disk.
        """
        key = self._cache_key(query)
        if key in self._cache:
            print(f"[Decomposer] Cache hit for: '{query[:50]}'")
            return self._cache[key]

        print(f"[Decomposer] Decomposing: '{query[:60]}'")
        t = time.time()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": _DECOMPOSE_PROMPT.format(query=query)},
            ],
            temperature=0.2,  # Low temperature for consistent structure
            max_tokens=300,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content.strip()
        elapsed = time.time() - t
        print(f"[Decomposer] LLM responded in {elapsed:.2f}s")

        sub_queries = self._parse_response(raw, query)
        print(f"[Decomposer] Got {len(sub_queries)} sub-queries:")
        for i, sq in enumerate(sub_queries):
            print(f"  [{i+1}] {sq}")

        self._cache[key] = sub_queries
        self._save_cache()
        return sub_queries

    def _parse_response(self, raw: str, original_query: str) -> list[str]:
        """Parse LLM JSON response into list of strings. Falls back to original query."""
        try:
            # Model may return {"questions": [...]} or just [...]
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                result = [str(q).strip() for q in parsed if str(q).strip()]
            elif isinstance(parsed, dict):
                # Find the first list value
                for v in parsed.values():
                    if isinstance(v, list):
                        result = [str(q).strip() for q in v if str(q).strip()]
                        break
                else:
                    result = [original_query]
            else:
                result = [original_query]

            # Sanity check: between 2 and 6 sub-queries
            if not (2 <= len(result) <= 6):
                return [original_query]
            return result
        except Exception as e:
            print(f"[Decomposer] JSON parse failed ({e}), using original query")
            return [original_query]

    # ------------------------------------------------------------------
    # Search interface
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        hybrid_search: HybridSearch,
        top_k: int = 20,
    ) -> tuple[list[SearchResult], list[str]]:
        """
        Decompose query -> search each sub-query -> aggregate & deduplicate.

        Strategy:
          - Run each sub-query independently through hybrid search.
          - Deduplicate by chunk_id, keeping the HIGHEST score seen.
          - Return top_k results sorted by max score.

        Returns:
            Tuple of (aggregated_results, sub_queries_used).
        """
        sub_queries = self.decompose(query)
        per_query_top_k = max(10, top_k // len(sub_queries))

        # Collect all results, deduplicate by chunk_id keeping best score
        best: dict[str, SearchResult] = {}

        for sq in sub_queries:
            sub_results = hybrid_search.search(sq, top_k=per_query_top_k)
            for r in sub_results:
                cid = r.chunk.chunk_id
                if cid not in best or r.score > best[cid].score:
                    best[cid] = r

        # Sort by score descending
        aggregated = sorted(best.values(), key=lambda r: r.score, reverse=True)
        return aggregated[:top_k], sub_queries
