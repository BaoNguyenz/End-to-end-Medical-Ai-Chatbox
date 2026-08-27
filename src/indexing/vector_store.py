"""
vector_store.py
Qdrant-based vector store with HNSW indexing for document chunks.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    HnswConfigDiff,
    PointStruct,
    SearchParams,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from src.models import Chunk, SearchResult, SearchSource


class VectorStore:
    """
    Manages a Qdrant collection with HNSW-indexed vectors.
    Handles embedding, indexing, and search of document chunks.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "enterprise_docs",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        hnsw_m: int = 32,
        hnsw_ef_construct: int = 200,
        hnsw_ef_search: int = 100,
        _embedding_model: Optional[SentenceTransformer] = None,
    ) -> None:
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construct = hnsw_ef_construct
        self.hnsw_ef_search = hnsw_ef_search
        self.model = _embedding_model or SentenceTransformer(embedding_model_name)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def create_collection(self, recreate: bool = False) -> None:
        """
        Create the Qdrant collection with HNSW configuration.
        If recreate=True, deletes existing collection first.
        """
        exists = self.client.collection_exists(self.collection_name)

        if exists and not recreate:
            print(f"[VectorStore] Collection '{self.collection_name}' already exists. Skipping.")
            return

        if exists and recreate:
            self.client.delete_collection(self.collection_name)
            print(f"[VectorStore] Deleted existing collection '{self.collection_name}'")

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_dim,
                distance=Distance.COSINE,
                hnsw_config=HnswConfigDiff(
                    m=self.hnsw_m,
                    ef_construct=self.hnsw_ef_construct,
                ),
            ),
        )
        print(
            f"[VectorStore] Created collection '{self.collection_name}' "
            f"(dim={self.embedding_dim}, HNSW m={self.hnsw_m}, "
            f"ef_construct={self.hnsw_ef_construct})"
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def index_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        """
        Encode and upsert chunks into Qdrant in batches.
        Uses chunk_id hash as point ID.
        """
        total = len(chunks)
        print(f"[VectorStore] Indexing {total} chunks (batch_size={batch_size})...")

        for start in range(0, total, batch_size):
            batch = chunks[start : start + batch_size]
            texts = [c.content for c in batch]
            embeddings = self._encode(texts)

            points = [
                PointStruct(
                    id=abs(hash(chunk.chunk_id)) % (2**63),  # Qdrant needs int or UUID
                    vector=embedding.tolist(),
                    payload=chunk.to_payload(),
                )
                for chunk, embedding in zip(batch, embeddings)
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            indexed = min(start + batch_size, total)
            print(f"  [{indexed}/{total}] chunks indexed")

        info = self.client.get_collection(self.collection_name)
        print(f"[VectorStore] Done. Collection has {info.points_count} vectors.")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 50,
        document_type: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Semantic search using query embedding.

        Args:
            query: Search query text.
            top_k: Number of results to return.
            document_type: Optional filter by document type.

        Returns:
            List of SearchResult ordered by score descending.
        """
        query_embedding = self._encode([query])[0].tolist()

        # Build optional filter
        query_filter = None
        if document_type:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_type",
                        match=MatchValue(value=document_type),
                    )
                ]
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            query_filter=query_filter,
            search_params=SearchParams(
                hnsw_ef=self.hnsw_ef_search,
                exact=False,
            ),
            with_payload=True,
        )

        results: list[SearchResult] = []
        for hit in response.points:
            payload = hit.payload or {}
            chunk = Chunk(
                content=payload.get("content", ""),
                chunk_id=payload.get("chunk_id", ""),
                doc_id=payload.get("doc_id", ""),
                document_type=payload.get("document_type", ""),
                chunk_index=payload.get("chunk_index", 0),
                metadata={
                    k: v for k, v in payload.items()
                    if k not in ("content", "chunk_id", "doc_id", "document_type", "chunk_index")
                },
            )
            results.append(SearchResult(
                chunk=chunk,
                score=hit.score,
                source=SearchSource.VECTOR,
            ))

        return results

    def search_by_embedding(
        self,
        embedding: list[float],
        top_k: int = 50,
        document_type: Optional[str] = None,
    ) -> list[SearchResult]:
        """Search using a pre-computed embedding vector (for HyDE)."""
        query_filter = None
        if document_type:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_type",
                        match=MatchValue(value=document_type),
                    )
                ]
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            limit=top_k,
            query_filter=query_filter,
            search_params=SearchParams(
                hnsw_ef=self.hnsw_ef_search,
                exact=False,
            ),
            with_payload=True,
        )

        results: list[SearchResult] = []
        for hit in response.points:
            payload = hit.payload or {}
            chunk = Chunk(
                content=payload.get("content", ""),
                chunk_id=payload.get("chunk_id", ""),
                doc_id=payload.get("doc_id", ""),
                document_type=payload.get("document_type", ""),
                chunk_index=payload.get("chunk_index", 0),
            )
            results.append(SearchResult(
                chunk=chunk,
                score=hit.score,
                source=SearchSource.VECTOR,
            ))

        return results

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_all_chunks(self) -> list[Chunk]:
        """Retrieve all chunks from the collection (for BM25 index building)."""
        chunks: list[Chunk] = []
        offset = None
        limit = 100

        while True:
            results, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in results:
                payload = point.payload or {}
                chunks.append(Chunk(
                    content=payload.get("content", ""),
                    chunk_id=payload.get("chunk_id", ""),
                    doc_id=payload.get("doc_id", ""),
                    document_type=payload.get("document_type", ""),
                    chunk_index=payload.get("chunk_index", 0),
                ))

            if next_offset is None:
                break
            offset = next_offset

        return chunks

    def collection_info(self) -> dict:
        """Get collection statistics."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "points_count": info.points_count,
            "status": info.status.value,
        }
