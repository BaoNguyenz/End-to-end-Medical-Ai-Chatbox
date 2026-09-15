"""
GaleMed AI — Langfuse Observability Tracker
=============================================
Provides LLM trace spans for the GaleMed RAG pipeline with:
    - Langfuse Cloud v4 integration (when API keys present)
    - Graceful MockTracer fallback (when keys missing / network unavailable)
    - PII masking before any data leaves the system
    - Context-manager API for clean span lifecycle management

Architecture:
    get_tracer()  ->  LangfuseTracer  (real, Langfuse v4 SDK)
                 ->  MockTracer       (fallback, zero external I/O)

Langfuse v4 API used:
    client.start_observation(name=..., as_type='span', input=..., metadata=...)
    client.update_current_span(output=..., metadata=...)
    client.set_current_trace_io(input=..., output=...)
    client.flush()

Usage:
    tracer = get_tracer()
    with tracer.trace("rag-query", user_id="anon") as root:
        with root.span("HybridSearch", input={"query": q}) as s:
            results = hybrid_search(q)
            s.set_output({"num_results": len(results)})
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import contextmanager
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)


# ── Sentinel for "langfuse not installed / not configured" ─────────────────────
_LANGFUSE_AVAILABLE = False
try:
    from langfuse import Langfuse  # type: ignore
    _LANGFUSE_AVAILABLE = True
except ImportError:
    logger.debug("langfuse package not installed — MockTracer will be used.")


# ── Span Context Object ────────────────────────────────────────────────────────

class SpanContext:
    """
    Thin wrapper around a Langfuse v4 span (or a no-op mock).

    Provides a unified API regardless of backend:
        span.set_output({"key": "value"})
        span.set_metadata({"cost_usd": 0.0012})
        span.set_level("ERROR")
    """

    def __init__(self, client: Any, name: str, is_mock: bool = False) -> None:
        self._client = client   # Langfuse client instance (used for update_current_span)
        self.name = name
        self.is_mock = is_mock
        self._start = time.perf_counter()
        self._output: Any = None
        self._metadata: dict = {}
        self._level: str = "DEFAULT"

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000

    def set_output(self, output: Any) -> None:
        self._output = output
        if not self.is_mock and self._client is not None:
            try:
                self._client.update_current_span(output=output)
            except Exception as exc:
                logger.debug("Langfuse set_output error (%s): %s", self.name, exc)

    def set_metadata(self, metadata: dict) -> None:
        self._metadata.update(metadata)
        if not self.is_mock and self._client is not None:
            try:
                self._client.update_current_span(metadata=metadata)
            except Exception as exc:
                logger.debug("Langfuse set_metadata error (%s): %s", self.name, exc)

    def set_level(self, level: str) -> None:
        """Set span level: 'DEFAULT' | 'DEBUG' | 'WARNING' | 'ERROR'"""
        self._level = level
        if not self.is_mock and self._client is not None:
            try:
                self._client.update_current_span(level=level)
            except Exception as exc:
                logger.debug("Langfuse set_level error (%s): %s", self.name, exc)

    def end(self) -> float:
        """Return elapsed milliseconds (span is ended by context manager exit)."""
        return self.elapsed_ms


class TraceContext:
    """
    Root trace context. Child spans are created via ``span()``.

    Example:
        with tracer.trace("rag-query") as root:
            with root.span("Retrieval") as s:
                docs = search(query)
                s.set_output({"count": len(docs)})
    """

    def __init__(
        self,
        client: Any,
        name: str,
        is_mock: bool = False,
        pii_masker: Any = None,
    ) -> None:
        self._client = client
        self.name = name
        self.is_mock = is_mock
        self._pii = pii_masker
        self.trace_id = str(uuid.uuid4())

    @contextmanager
    def span(
        self,
        name: str,
        input: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> Generator[SpanContext, None, None]:
        """
        Context manager for a child observation span (Langfuse v4).

        Args:
            name:     Human-readable span name (e.g. "HybridSearch")
            input:    Input payload — will be PII-masked before upload.
            metadata: Extra key-value metadata attached to the span.

        Yields:
            SpanContext — call .set_output(), .set_metadata() inside the block.
        """
        safe_input = self._pii.mask_dict(input) if (self._pii and input) else input
        safe_meta  = self._pii.mask_dict(metadata) if (self._pii and metadata) else metadata

        ctx = SpanContext(client=self._client, name=name, is_mock=self.is_mock)

        if not self.is_mock and self._client is not None:
            try:
                with self._client.start_as_current_observation(
                    name=name,
                    as_type="span",
                    input=safe_input,
                    metadata=safe_meta,
                ):
                    try:
                        yield ctx
                    finally:
                        elapsed = ctx.end()
                        logger.debug("[LangfuseTracer] span=%s elapsed=%.1fms", name, elapsed)
                return
            except Exception as exc:
                logger.debug("Langfuse span error (%s): %s — using mock", name, exc)
                ctx.is_mock = True

        # Mock path
        try:
            yield ctx
        finally:
            elapsed = ctx.end()
            logger.debug("[MockSpan] span=%s elapsed=%.1fms", name, elapsed)

    def update(self, output: Any = None, metadata: Optional[dict] = None) -> None:
        """Update the root trace with final output/metadata."""
        if not self.is_mock and self._client is not None:
            try:
                kwargs: dict = {}
                if output is not None:
                    kwargs["output"] = output
                if metadata is not None:
                    # set_current_trace_io only takes input/output; metadata goes via update
                    pass
                self._client.set_current_trace_io(**kwargs)
            except Exception as exc:
                logger.debug("Langfuse trace.update() error: %s", exc)


# ── Tracer Classes ─────────────────────────────────────────────────────────────

class MockTracer:
    """
    Zero-dependency, no-I/O tracer used as Graceful Fallback.

    Activated automatically when:
        - langfuse package is not installed, OR
        - LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY env vars are missing.

    All methods are safe no-ops — the pipeline continues unaffected.
    """

    @contextmanager
    def trace(
        self,
        name: str,
        user_id: str = "anon",
        session_id: Optional[str] = None,
        input: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> Generator[TraceContext, None, None]:
        logger.debug("[MockTracer] trace=%s user=%s (no Langfuse connection)", name, user_id)
        ctx = TraceContext(client=None, name=name, is_mock=True, pii_masker=None)
        yield ctx

    def flush(self) -> None:
        pass


class LangfuseTracer:
    """
    Production Langfuse Cloud tracer (Langfuse SDK v4).

    Uses start_as_current_observation() context manager API.
    All query/answer content is PII-masked before leaving the system.
    """

    def __init__(
        self,
        public_key: str,
        secret_key: str,
        host: str = "https://cloud.langfuse.com",
    ) -> None:
        if not _LANGFUSE_AVAILABLE:
            raise RuntimeError("langfuse package is not installed. Run: pip install langfuse")

        self._client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )

        from src.observability.pii_masker import PIIMasker  # pyrefly: ignore [missing-import]
        self._pii = PIIMasker()

        logger.info("LangfuseTracer connected to %s", host)

    @contextmanager
    def trace(
        self,
        name: str,
        user_id: str = "anon",
        session_id: Optional[str] = None,
        input: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> Generator[TraceContext, None, None]:
        """
        Open a root trace for a single request (Langfuse v4).

        Uses start_as_current_observation() as the root span, then sets
        trace IO after the block completes.
        """
        safe_input = self._pii.mask_dict(input) if input else None
        safe_meta  = self._pii.mask_dict(metadata) if metadata else None

        ctx = TraceContext(
            client=self._client,
            name=name,
            is_mock=False,
            pii_masker=self._pii,
        )

        try:
            with self._client.start_as_current_observation(
                name=name,
                as_type="span",
                input=safe_input,
                metadata=safe_meta,
            ):
                # Capture current trace ID if available
                try:
                    tid = self._client.get_current_trace_id()
                    if tid:
                        ctx.trace_id = tid
                except Exception:
                    pass

                try:
                    yield ctx
                finally:
                    self.flush()
        except Exception as exc:
            logger.warning("LangfuseTracer root trace failed: %s — yielding mock context", exc)
            mock_ctx = TraceContext(client=None, name=name, is_mock=True, pii_masker=None)
            yield mock_ctx

    def flush(self) -> None:
        """Flush pending spans to Langfuse Cloud."""
        try:
            self._client.flush()
        except Exception as exc:
            logger.debug("Langfuse flush error: %s", exc)


# ── Singleton Factory ──────────────────────────────────────────────────────────

_tracer_instance: Optional[LangfuseTracer | MockTracer] = None


def get_tracer(force_mock: bool = False) -> LangfuseTracer | MockTracer:
    """
    Return the singleton tracer instance.

    Decision logic:
        1. If force_mock=True                          -> MockTracer
        2. If langfuse not installed                   -> MockTracer (warning)
        3. If LANGFUSE_PUBLIC_KEY / SECRET missing     -> MockTracer (warning)
        4. Otherwise                                   -> LangfuseTracer (Cloud)

    Example:
        tracer = get_tracer()
        with tracer.trace("rag-query", user_id="session-xyz") as root:
            with root.span("LLMGeneration", input={"query": q}) as s:
                answer = call_llm(q)
                s.set_output({"answer_length": len(answer)})
    """
    global _tracer_instance

    if _tracer_instance is not None:
        return _tracer_instance

    if force_mock or not _LANGFUSE_AVAILABLE:
        if not _LANGFUSE_AVAILABLE:
            logger.warning(
                "langfuse not installed — observability disabled. "
                "Install with: pip install langfuse"
            )
        _tracer_instance = MockTracer()
        return _tracer_instance

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key  = os.getenv("LANGFUSE_SECRET_KEY", "")
    host        = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        logger.warning(
            "LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY not set. "
            "Using MockTracer — set these in .env to enable full observability."
        )
        _tracer_instance = MockTracer()
        return _tracer_instance

    try:
        _tracer_instance = LangfuseTracer(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        logger.info("Langfuse Cloud observability enabled.")
    except Exception as exc:
        logger.warning("LangfuseTracer init failed (%s) — falling back to MockTracer.", exc)
        _tracer_instance = MockTracer()

    return _tracer_instance


def reset_tracer() -> None:
    """Reset the singleton (used in testing to force re-initialisation)."""
    global _tracer_instance
    _tracer_instance = None
