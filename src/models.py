"""
models.py
Shared data models used across all modules.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DocumentType(str, Enum):
    TECHNICAL = "technical"
    POLICY = "policy"
    PRODUCT = "product"


class SearchStrategy(str, Enum):
    BM25 = "bm25"
    VECTOR = "vector"
    HYBRID = "hybrid"


class QueryType(str, Enum):
    SIMPLE = "simple"
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    COMPLEX = "complex"
    VAGUE = "vague"


class SearchSource(str, Enum):
    BM25 = "bm25"
    VECTOR = "vector"
    HYBRID = "hybrid"
    GRAPH = "graph"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """Raw document loaded from disk."""

    page_content: str
    metadata: dict = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        return self.metadata.get("doc_id", "")

    @property
    def document_type(self) -> str:
        return self.metadata.get("document_type", "")

    @property
    def filename(self) -> str:
        return self.metadata.get("filename", "")


@dataclass
class Chunk:
    """A semantically coherent piece of a document."""

    content: str
    chunk_id: str
    doc_id: str
    document_type: str
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = field(default=None, repr=False)

    def to_payload(self) -> dict:
        """Convert to Qdrant payload (excludes embedding)."""
        return {
            "content": self.content,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "document_type": self.document_type,
            "chunk_index": self.chunk_index,
            **self.metadata,
        }


@dataclass
class SearchResult:
    """A single search result with score and provenance."""

    chunk: Chunk
    score: float
    source: SearchSource

    def __repr__(self) -> str:
        preview = self.chunk.content[:80].replace("\n", " ")
        return (
            f"SearchResult(score={self.score:.4f}, "
            f"source={self.source.value}, "
            f"doc={self.chunk.doc_id}, "
            f'"{preview}...")'
        )


@dataclass
class RAGResponse:
    """Final response from the RAG pipeline."""

    query: str
    answer: str
    sources: list[SearchResult] = field(default_factory=list)
    latency: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
