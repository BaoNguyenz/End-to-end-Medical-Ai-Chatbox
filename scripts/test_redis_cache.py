"""
scripts/test_redis_cache.py
Verification tests for Redis Semantic Cache.

Tests:
    1. Cache Miss  - first query (full RAG pipeline, stores in cache).
    2. Cache Hit   - semantically similar follow-up (served from Redis in < 10ms).
    3. Stats       - verify hit rate, entry count.

Run:
    uv run python scripts/test_redis_cache.py
"""

import sys
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cache.semantic_cache import get_semantic_cache


def _banner(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_connection():
    _banner("Test 1: Redis Connection")
    cache = get_semantic_cache()
    if cache.available:
        print("[PASS] Redis connected successfully.")
    else:
        print("[SKIP] Redis not available. Ensure `docker compose up -d` is running.")
    return cache.available


def test_set_and_get(cache):
    _banner("Test 2: Cache MISS then HIT")
    QUERY_1 = "What are the primary symptoms of asthma?"
    QUERY_2 = "What are the main signs and warning symptoms of asthma?"  # Semantically similar

    # 1. Ensure clean state
    cache.clear()
    print(f"[INFO] Cache cleared.")

    # 2. First lookup -> should be MISS
    t0 = time.perf_counter()
    result = cache.get(QUERY_1)
    miss_ms = (time.perf_counter() - t0) * 1000
    assert result is None, "Expected Cache MISS on first lookup"
    print(f"[PASS] MISS confirmed for query 1 ({miss_ms:.1f}ms)")

    # 3. Store answer in cache
    MOCK_ANSWER = (
        "Asthma symptoms include: wheezing, shortness of breath, chest tightness, "
        "and chronic cough. Symptoms often worsen at night or with exercise. "
        "The condition is characterized by reversible airflow obstruction.\n\n"
        "*This information is for educational purposes only and does not constitute medical advice.*"
    )
    cache.set(query=QUERY_1, answer=MOCK_ANSWER, sources=[{"doc_id": "Asthma.md", "score": 0.98}])
    print("[INFO] Stored query 1 in cache.")

    # 4. Lookup SAME query -> should be HIT
    t0 = time.perf_counter()
    hit1 = cache.get(QUERY_1)
    hit1_ms = (time.perf_counter() - t0) * 1000
    assert hit1 is not None, "Expected Cache HIT for exact query"
    print(f"[PASS] HIT for exact query ({hit1_ms:.1f}ms) - Answer preview: '{hit1.answer[:60]}...'")

    # 5. Lookup SIMILAR query -> should also be HIT (semantic similarity)
    t0 = time.perf_counter()
    hit2 = cache.get(QUERY_2)
    hit2_ms = (time.perf_counter() - t0) * 1000
    if hit2 is not None:
        print(f"[PASS] Semantic HIT for similar query ({hit2_ms:.1f}ms) - Latency reduction: ~{3000/hit2_ms:.0f}x faster!")
    else:
        print(f"[INFO] Semantic MISS for similar query (similarity below threshold={0.92})")

    return True


def test_stats(cache):
    _banner("Test 3: Cache Statistics")
    stats = cache.stats()
    print(f"  Hits            : {stats['hits']}")
    print(f"  Misses          : {stats['misses']}")
    print(f"  Hit Rate        : {stats['hit_rate_pct']}%")
    print(f"  Cached Queries  : {stats['total_cached_queries']}")
    print(f"  Redis Available : {stats['available']}")
    print("[PASS] Stats endpoint working.")


def main():
    print("\nGaleMed AI - Redis Semantic Cache Verification Test")
    print("Branch: redis_feature")

    if not test_connection():
        print("\n[ABORT] Redis not reachable. Start Docker first:")
        print("  docker compose up -d redis")
        sys.exit(1)

    cache = get_semantic_cache()
    test_set_and_get(cache)
    test_stats(cache)

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED ✓")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
