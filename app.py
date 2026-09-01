"""
app.py
FastAPI backend for the Enterprise RAG system.

Endpoints:
  POST /api/query   - Main query endpoint
  GET  /api/health  - Health check (Qdrant + Neo4j status)
  GET  /api/stats   - Collection statistics
  GET  /            - Serve frontend
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# pyrefly: ignore [missing-import]
from src.config import settings
# pyrefly: ignore [missing-import]
from src.orchestrator.pipeline import RAGPipeline

# ── Lifespan: initialize pipeline on startup ───────────────────────────────

pipeline: Optional[RAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    print("Starting up RAG pipeline...")
    try:
        pipeline = RAGPipeline(use_graph=True)
    except Exception as e:
        print(f"[WARN] Pipeline init error (graph may be unavailable): {e}")
        pipeline = RAGPipeline(use_graph=False)
    yield
    print("Shutting down.")


# ── App ────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Medical Generative AI Chatbot",
    description="Advanced RAG with Hybrid Search, GraphRAG, and Query Transformation",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    search_mode: str = "auto"   # auto | hybrid | vector | bm25 | graph
    top_k: int = 10
    use_graph: Optional[bool] = None


class SourceItem(BaseModel):
    doc_id: str
    chunk_id: str
    score: float
    source: str
    content_preview: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceItem]
    latency: dict
    metadata: dict


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        response = pipeline.process_query(
            query=req.query,
            search_mode=req.search_mode,
            top_k=req.top_k,
            use_graph=req.use_graph,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    sources = [
        SourceItem(
            doc_id=r.chunk.doc_id,
            chunk_id=r.chunk.chunk_id,
            score=round(r.score, 4),
            source=r.source.value,
            content_preview=r.chunk.content[:200],
        )
        for r in response.sources
    ]

    return QueryResponse(
        query=response.query,
        answer=response.answer,
        sources=sources,
        latency={k: round(v * 1000, 1) for k, v in response.latency.items()},  # ms
        metadata=response.metadata,
    )


@app.get("/api/health")
async def health():
    status: dict = {"status": "ok", "components": {}}

    # Qdrant
    try:
        info = pipeline.vector_store.collection_info() if pipeline else {}
        status["components"]["qdrant"] = {
            "status": "ok",
            "collection": info.get("name"),
            "points": info.get("points_count"),
        }
    except Exception as e:
        status["components"]["qdrant"] = {"status": "error", "detail": str(e)}

    # Neo4j
    try:
        if pipeline and pipeline.graph_retriever:
            counts = pipeline.graph_retriever.kg.get_node_counts()
            status["components"]["neo4j"] = {"status": "ok", "counts": counts}
        else:
            status["components"]["neo4j"] = {"status": "disabled"}
    except Exception as e:
        status["components"]["neo4j"] = {"status": "error", "detail": str(e)}

    # BM25
    if pipeline:
        status["components"]["bm25"] = {
            "status": "ok",
            "corpus_size": pipeline.bm25.corpus_size,
        }

    return status


@app.get("/api/stats")
async def stats():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    info = pipeline.vector_store.collection_info()
    result: dict = {
        "vector_store": info,
        "bm25_corpus_size": pipeline.bm25.corpus_size,
        "graph_enabled": pipeline.graph_retriever is not None,
    }

    if pipeline.graph_retriever:
        try:
            result["graph_counts"] = pipeline.graph_retriever.kg.get_node_counts()
        except Exception:
            result["graph_counts"] = {}

    return result


# ── Frontend static files ──────────────────────────────────────────────────

# ── Frontend static files & assets ──────────────────────────────────────────


# ── Streaming Query Endpoint (SSE) ────────────────────────────────────────────

class StreamQueryRequest(QueryRequest):
    """Same as QueryRequest; processed via streaming SSE."""
    pass


@app.post("/api/query/stream")
async def query_stream_endpoint(req: StreamQueryRequest):
    """
    Stream medical AI answer tokens via Server-Sent Events (SSE).

    Frontend reads: response.body.getReader() with TextDecoder.
    Each event: data: {"type": "token"|"done"|"error", ...}\n\n
    """
    if pipeline is None:
        async def _err():
            yield b'data: {"type":"error","message":"Pipeline not initialized"}\n\n'
        return StreamingResponse(_err(), media_type="text/event-stream")

    async def _generate() -> AsyncGenerator[bytes, None]:
        try:
            for chunk in pipeline.process_query_stream(
                query=req.query,
                search_mode=req.search_mode,
                top_k=req.top_k,
                use_graph=req.use_graph,
            ):
                event = "data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n"
                yield event.encode("utf-8")
        except Exception as e:
            err_event = f'data: {{"type":"error","message":"{str(e)}"}}\n\n'
            yield err_event.encode("utf-8")

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # Disable nginx buffering for SSE
            "Connection": "keep-alive",
        },
    )


# ── Redis Semantic Cache Endpoints ─────────────────────────────────────────────

@app.get("/api/cache/stats")
async def cache_stats():
    """Return Redis semantic cache performance statistics."""
    try:
        # pyrefly: ignore [missing-import]
        from src.cache.semantic_cache import get_semantic_cache
        cache = get_semantic_cache()
        return {"status": "ok", "cache": cache.stats()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.delete("/api/cache/clear")
async def cache_clear():
    """Clear all semantic cache entries."""
    try:
        # pyrefly: ignore [missing-import]
        from src.cache.semantic_cache import get_semantic_cache
        cache = get_semantic_cache()
        deleted = cache.clear()
        return {"status": "ok", "deleted": deleted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


import os
_DIST = os.path.join(os.path.dirname(__file__), "frontend", "dist")
_FRONTEND_DIR = _DIST if os.path.isdir(_DIST) else os.path.join(os.path.dirname(__file__), "frontend")
_ASSETS_DIR = os.path.join(_FRONTEND_DIR, "assets")

if os.path.isdir(_ASSETS_DIR):
    app.mount("/assets", StaticFiles(directory=_ASSETS_DIR), name="assets")

if os.path.isdir(_FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=_FRONTEND_DIR), name="static")

    @app.get("/")
    async def serve_frontend():
        index_file = os.path.join(_FRONTEND_DIR, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"message": "Medical AI API is running", "docs": "/docs"}
else:
    @app.get("/")
    async def root():
        return {"message": "Medical AI API", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=False)
