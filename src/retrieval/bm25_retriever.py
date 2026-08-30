"""
bm25_retriever.py
BM25 keyword-based retrieval using Okapi BM25.
"""

from __future__ import annotations

import re
import string

import nltk
from rank_bm25 import BM25Okapi

from src.models import Chunk, SearchResult, SearchSource


# Ensure punkt tokenizer is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


class BM25Retriever:
    """BM25 keyword retriever over document chunks."""

    def __init__(self) -> None:
        self._index: BM25Okapi | None = None
        self._chunks: list[Chunk] = []
        self._tokenized_corpus: list[list[str]] = []

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Tokenize text for BM25: lowercase, remove punctuation, split.
        Uses nltk.word_tokenize for consistent tokenization.
        """
        text = text.lower()
        tokens = nltk.word_tokenize(text)
        # Remove pure-punctuation tokens but keep alphanumeric tokens
        tokens = [
            t for t in tokens
            if not all(c in string.punctuation for c in t)
        ]
        return tokens

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build_index(self, chunks: list[Chunk]) -> None:
        """
        Build BM25 index from a list of chunks.

        Args:
            chunks: List of Chunk objects to index.
        """
        self._chunks = chunks
        self._tokenized_corpus = [
            self._tokenize(chunk.content) for chunk in chunks
        ]
        self._index = BM25Okapi(self._tokenized_corpus)
        print(f"[BM25] Built index over {len(chunks)} chunks")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 50) -> list[SearchResult]:
        """
        Search using BM25 scoring.

        Args:
            query: The search query.
            top_k: Number of results to return.

        Returns:
            List of SearchResult sorted by BM25 score descending.
        """
        if self._index is None:
            raise RuntimeError("BM25 index not built. Call build_index() first.")

        tokenized_query = self._tokenize(query)
        scores = self._index.get_scores(tokenized_query)

        # Get top-k indices sorted by score
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results: list[SearchResult] = []
        for idx in top_indices:
            if scores[idx] > 0:  # Skip zero-score results
                results.append(SearchResult(
                    chunk=self._chunks[idx],
                    score=float(scores[idx]),
                    source=SearchSource.BM25,
                ))

        return results

    @property
    def is_built(self) -> bool:
        return self._index is not None

    @property
    def corpus_size(self) -> int:
        return len(self._chunks)
