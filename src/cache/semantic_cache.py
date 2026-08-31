"""
src/cache/semantic_cache.py
Redis Semantic Cache using RediSearch HNSW Vector Index.

Flow:
    1. Embed query -> 384-dim vector (same model as retrieval: all-MiniLM-L6-v2).
    2. KNN search in Redis (HNSW index, cosine distance).
    3. If distance <= (1 - threshold): return cached answer (Cache HIT ~2-5ms).
    4. Otherwise: run full RAG pipeline, then store result (Cache MISS).
"""

from __future__ import annotations

import json
import logging
import struct
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import redis
from redis.commands.search.field import TagField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer

from src.config import settings

logger = logging.getLogger(__name__)

# Constants
_INDEX_NAME = "semantic_cache_idx"
_PREFIX = "medical_cache:"
_VECTOR_DIM = 384
_DISTANCE_METRIC = "COSINE"


@dataclass
class CacheEntry:
    """Represents a single cache record."""
    query_text: str
    answer: str
    sources: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    hit_count: int = 0


class RedisSemanticCache:
    """
    Semantic cache backed by Redis RediSearch HNSW Vector Index.
    - Embedding model: all-MiniLM-L6-v2 (RAM-based, no GPU/VRAM).
    - Storage: RAM-based, capped at 512MB by Redis maxmemory config.
    - Eviction: allkeys-lru (oldest entries removed automatically when full).
    """

    def __init__(self) -> None:
        self._client = self._connect()
        self._encoder = None
        self._stats = {"hits": 0, "misses": 0, "sets": 0, "errors": 0}
        if self._client:
            self._ensure_index()

    def _connect(self):
        """Establish connection to Redis. Returns None if unavailable."""
        try:
            client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                decode_responses=False,
                socket_connect_timeout=2,
                socket_timeout=2,
                retry_on_timeout=False,
            )
            client.ping()
            logger.info("[SemanticCache] Connected to Redis at %s:%d",
                        settings.redis_host, settings.redis_port)
            return client
        except Exception as e:
            logger.warning("[SemanticCache] Redis unavailable: %s (cache disabled)", e)
            return None

    def _ensure_index(self) -> None:
        """Create the HNSW Vector index if not already present."""
        try:
            self._client.ft(_INDEX_NAME).info()
            logger.info("[SemanticCache] RediSearch index already exists.")
        except Exception:
            schema = (
                TextField("$.query_text", as_name="query_text"),
                TagField("$.cache_id", as_name="cache_id"),
                VectorField(
                    "$.query_vector", "HNSW",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": _VECTOR_DIM,
                        "DISTANCE_METRIC": _DISTANCE_METRIC,
                        "M": 16,
                        "EF_CONSTRUCTION": 200,
                        "EF_RUNTIME": 100,
                    },
                    as_name="query_vector",
                ),
            )
            self._client.ft(_INDEX_NAME).create_index(
                schema,
                definition=IndexDefinition(prefix=[_PREFIX], index_type=IndexType.JSON),
            )
            logger.info("[SemanticCache] Created RediSearch HNSW index.")

    def _encode(self, text: str) -> np.ndarray:
        """Encode query text to 384-dim float32 vector (CPU-only)."""
        if self._encoder is None:
            self._encoder = SentenceTransformer(settings.embedding_model)
        return self._encoder.encode(
            text, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)

    @staticmethod
    def _vec_to_bytes(vec: np.ndarray) -> bytes:
        return struct.pack(f"{len(vec)}f", *vec)

    def get(self, query: str):
        """
        Search for a semantically similar cached answer.
        Returns CacheEntry if similarity >= threshold, else None.
        Time: ~2-5ms on Cache HIT.

        Note: redis-py v8 Document fields are accessed as attributes (doc.field_name)
        using the as_name alias set in return_fields(). Dict-style access is not supported.
        """
        if not self._client:
            return None

        t0 = time.perf_counter()
        try:
            vec = self._encode(query)
            vec_bytes = self._vec_to_bytes(vec)

            # Return fields using as_name aliases (no "$." prefix in attribute names)
            q = (
                Query("*=>[KNN 1 @query_vector $vec AS dist]")
                .sort_by("dist")
                .return_fields("query_text", "answer", "sources", "metadata",
                               "hit_count", "dist")
                .dialect(2)
            )
            results = self._client.ft(_INDEX_NAME).search(
                q, query_params={"vec": vec_bytes}
            )

            if not results.docs:
                self._stats["misses"] += 1
                return None

            doc = results.docs[0]

            # redis-py v8: Document fields accessed as attributes, not dict keys
            raw_dist = getattr(doc, "dist", "2.0")
            distance = float(raw_dist) if raw_dist is not None else 2.0
            similarity = 1.0 - distance

            if similarity < settings.semantic_cache_threshold:
                self._stats["misses"] += 1
                logger.debug("[SemanticCache] MISS (sim=%.3f) | %.1fms",
                             similarity, (time.perf_counter() - t0) * 1000)
                return None

            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._stats["hits"] += 1
            logger.info("[SemanticCache] HIT (sim=%.3f) | %.1fms", similarity, elapsed_ms)

            # Refresh TTL and increment hit counter (best-effort)
            try:
                key = doc.id
                raw = self._client.json().get(key, "$.hit_count")
                old_count = int(raw[0]) if raw else 0
                self._client.json().set(key, "$.hit_count", old_count + 1)
                self._client.expire(key, settings.semantic_cache_ttl)
            except Exception:
                pass

            # Access via attribute (redis-py v8 Document API)
            query_text = getattr(doc, "query_text", "")
            answer = getattr(doc, "answer", "")
            sources_raw = getattr(doc, "sources", "[]")
            meta_raw = getattr(doc, "metadata", "{}")
            hit_count_raw = getattr(doc, "hit_count", "0")

            return CacheEntry(
                query_text=query_text or "",
                answer=answer or "",
                sources=json.loads(sources_raw) if isinstance(sources_raw, str) else [],
                metadata=json.loads(meta_raw) if isinstance(meta_raw, str) else {},
                latency_ms=elapsed_ms,
                hit_count=int(hit_count_raw or 0) + 1,
            )

        except Exception as e:
            self._stats["errors"] += 1
            logger.warning("[SemanticCache] GET error: %s", e)
            return None

    def set(self, query: str, answer: str, sources=None, metadata=None) -> bool:
        """
        Store a new query-answer pair in Redis with TTL.
        Returns True on success, False on error.
        """
        if not self._client:
            return False

        try:
            vec = self._encode(query)
            cache_id = str(uuid.uuid4())
            key = f"{_PREFIX}{cache_id}"

            doc = {
                "cache_id": cache_id,
                "query_text": query,
                "answer": answer,
                "sources": json.dumps(sources or []),
                "metadata": json.dumps(metadata or {}),
                "query_vector": vec.tolist(),
                "hit_count": 0,
            }

            pipe = self._client.pipeline()
            pipe.json().set(key, "$", doc)
            pipe.expire(key, settings.semantic_cache_ttl)
            pipe.execute()

            self._stats["sets"] += 1
            logger.debug("[SemanticCache] SET query (%.60s)", query)
            return True

        except Exception as e:
            self._stats["errors"] += 1
            logger.warning("[SemanticCache] SET error: %s", e)
            return False

    def stats(self) -> dict:
        """Return cache performance statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0.0
        entry_count = 0
        if self._client:
            try:
                info = self._client.ft(_INDEX_NAME).info()
                entry_count = int(info.get("num_docs", 0))
            except Exception:
                pass
        return {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "sets": self._stats["sets"],
            "errors": self._stats["errors"],
            "hit_rate_pct": round(hit_rate * 100, 1),
            "total_cached_queries": entry_count,
            "available": self._client is not None,
        }

    def clear(self) -> int:
        """Delete all cached entries. Returns number of keys deleted."""
        if not self._client:
            return 0
        try:
            keys = self._client.keys(f"{_PREFIX}*")
            if keys:
                deleted = self._client.delete(*keys)
                logger.info("[SemanticCache] Cleared %d entries.", deleted)
                return int(deleted)
            return 0
        except Exception as e:
            logger.warning("[SemanticCache] CLEAR error: %s", e)
            return 0

    @property
    def available(self) -> bool:
        return self._client is not None


# Module-level singleton
_cache_instance = None


def get_semantic_cache() -> RedisSemanticCache:
    """Get or create the singleton cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = RedisSemanticCache()
    return _cache_instance
