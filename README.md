# 🩺 GaleMed AI — Clinical GraphRAG & Hybrid Search Intelligence System

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.13" />
  <img src="https://img.shields.io/badge/FastAPI-0.141-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/React-19_Vite-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React 19" />
  <img src="https://img.shields.io/badge/Qdrant-Vector_DB-DC382D?style=for-the-badge&logo=qdrant&logoColor=white" alt="Qdrant" />
  <img src="https://img.shields.io/badge/Neo4j-GraphRAG-008CC1?style=for-the-badge&logo=neo4j&logoColor=white" alt="Neo4j" />
  <img src="https://img.shields.io/badge/Redis_Stack-Semantic_Cache-DC382D?style=for-the-badge&logo=redis&logoColor=white" alt="Redis Stack" />
  <img src="https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI" />
</p>

---

## 📌 Executive Summary

**GaleMed AI** is a production-grade **Medical Decision-Support & Clinical Intelligence Engine** grounded strictly in **The Gale Encyclopedia of Medicine (3rd Edition)**.

By combining **Hybrid Multi-Vector Retrieval**, **Neo4j Knowledge Graph Traversal (GraphRAG)**, **Cross-Encoder Reranking**, **Redis HNSW Semantic Caching**, and **Server-Sent Events (SSE) Token Streaming**, GaleMed AI delivers accurate, verifiable, and low-latency clinical synthesis while eliminating hallucinations and protecting API budgets.

---

## 🏛️ System Architecture

```
                                  ┌───────────────────────────┐
                                  │   User / Clinical Client  │
                                  │   (React 19 + Vite Web)   │
                                  └─────────────┬─────────────┘
                                                │ HTTP / SSE Stream (:8080)
                                                ▼
                                  ┌───────────────────────────┐
                                  │     FastAPI Application   │
                                  │  (app.py / RAG Pipeline)  │
                                  └─────────────┬─────────────┘
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 │                                                             │
                 ▼                                                             ▼
    ┌──────────────────────────┐                                  ┌──────────────────────────┐
    │  🚨 Emergency Detector   │ ── (Life-threatening?) ────────► │ Instant Emergency Advice │
    └────────────┬─────────────┘                                  └──────────────────────────┘
                 │ (Normal Query)
                 ▼
    ┌──────────────────────────┐
    │  Redis Semantic Cache    │ ── (Cosine Sim ≥ 92%) ─────────► ⚡ Cache HIT (< 10ms)
    │  (HNSW 384-dim Index)    │
    └────────────┬─────────────┘
                 │ (Cache MISS)
                 ▼
    ┌──────────────────────────┐
    │   Query & Transformation │
    │   Router (HyDE / Decomp) │
    └────────────┬─────────────┘
                 │
       ┌─────────┴───────────────────────────────┐
       │                                         │
       ▼                                         ▼
┌───────────────┐                         ┌───────────────┐
│ Dense Vector  │                         │ Sparse Search │
│    Qdrant     │                         │     BM25      │
│ (13,350 Pts)  │                         │ (13,350 Pts)  │
└───────┬───────┘                         └───────┬───────┘
        │                                         │
        └────────────────┬────────────────────────┘
                         │ Reciprocal Rank Fusion (RRF)
                         ▼
        ┌────────────────────────────────┐
        │   Neo4j Clinical GraphRAG      │ ◄── Entities & Multi-hop Relations
        │ (Diseases, Meds, Symptoms)     │     (288 Diseases, 333 Meds, 508 Symptoms)
        └────────────────┬───────────────┘
                         │ Merge Candidates
                         ▼
        ┌────────────────────────────────┐
        │   Cross-Encoder Reranker       │
        │ (ms-marco-MiniLM-L-6-v2) + MMR │
        └────────────────┬───────────────┘
                         │ Top Context Chunks
                         ▼
        ┌────────────────────────────────┐
        │   OpenAI GPT-4o-mini           │ ──► Store answer in Redis Semantic Cache
        │   Streaming Generator (SSE)    │
        └────────────────┬───────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │   Verified Medical Answer      │
        │   + Citations & Disclaimers    │
        └────────────────────────────────┘
```

---

## ⚡ Key Highlights & Capabilities

### 1. 🔀 Multi-Stage Hybrid Search & GraphRAG
* **Dense Semantic Matching**: Qdrant vector database hosting 13,350 chunk embeddings generated via `sentence-transformers/all-MiniLM-L6-v2`.
* **Sparse Keyword Matching**: Custom rank-BM25 retriever tuned for exact medical terms, drug brands, and anatomical names.
* **Knowledge Graph Traversal**: Neo4j GraphRAG capturing structured relations across **288 Diseases**, **333 Medications**, **508 Symptoms**, **99 Procedures**, and **441 explicit relationships** (e.g. `TREATS`, `CAUSES`, `CONTRAINDICATED_WITH`).

### 2. ⚡ Sub-10ms Redis Semantic Caching
* Backed by **Redis Stack Server** using an in-memory **HNSW Vector Index** (384 dimensions, Cosine distance).
* Semantically similar questions (similarity $\ge 92\%$) are answered instantly from RAM in **< 10ms**, bypassing all retrieval layers and reducing OpenAI API costs to **$0**.

### 3. 🌊 Real-Time Token Streaming (SSE)
* Implemented via FastAPI `StreamingResponse` using Server-Sent Events (`text/event-stream`).
* Time-to-First-Token (TTFT) reduced to **~250ms**, providing a smooth, typewriter-style reading experience for clinicians.

### 4. 🛡️ Clinical Safety Guardrails
* **Emergency Reflex Engine**: Identifies critical symptoms (e.g., crushing chest pain, anaphylaxis, acute stroke) and returns immediate triage instructions.
* **Grounding & Disclaimers**: Answers are constrained strictly to context from *The Gale Encyclopedia of Medicine*, accompanied by mandatory educational disclaimers.

---

## 📂 Project Structure

```
Medical_Generative_AI_v2/
├── .github/
│   └── workflows/
│       └── deploy.yml              # CI/CD automation for AWS EC2
├── Data/
│   └── gale_encyclopedia_data/     # Structured encyclopedia source documents
├── cache/                          # Local BM25 persisted corpus & embeddings
├── frontend/                       # React 19 + Vite modern client
│   ├── src/
│   │   ├── App.jsx                 # Interactive Chatbot UI + Telemetry Panel
│   │   ├── main.jsx
│   │   └── index.css
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── src/                            # Core Python RAG backend package
│   ├── cache/
│   │   └── semantic_cache.py       # Redis RediSearch HNSW semantic cache engine
│   ├── graph/
│   │   ├── entity_extractor.py     # Medical entity extraction pipeline
│   │   ├── entity_models.py        # Pydantic entity schema models
│   │   ├── graph_retriever.py      # Neo4j NL-to-Cypher search retriever
│   │   └── knowledge_graph.py      # Neo4j driver & graph builder
│   ├── indexing/
│   │   ├── document_loader.py      # Medical document parsing & cleaning
│   │   ├── semantic_chunker.py     # Section-aware medical chunking
│   │   └── vector_store.py         # Qdrant client wrapper
│   ├── orchestrator/
│   │   ├── evaluator.py            # Automated RAG benchmarking harness
│   │   └── pipeline.py             # Main end-to-end RAG pipeline & streaming
│   ├── post_retrieval/
│   │   ├── cross_encoder_reranker.py # ms-marco-MiniLM-L-6-v2 CrossEncoder
│   │   ├── mmr.py                  # Maximal Marginal Relevance diversification
│   │   └── post_retrieval_pipeline.py
│   ├── retrieval/
│   │   ├── bm25_retriever.py       # BM25 sparse keyword retriever
│   │   ├── hybrid_search.py        # Reciprocal Rank Fusion engine
│   │   └── query_router.py         # Intent classifier (Direct/HyDE/Decompose)
│   ├── transformation/
│   │   ├── hyde.py                 # Hypothetical Document Embeddings
│   │   ├── query_decomposition.py  # Multi-hop sub-query decomposition
│   │   └── transformation_router.py
│   ├── config.py                   # Pydantic environment configuration
│   └── models.py                   # Shared data contracts & DTOs
├── scripts/                        # Utility & operational automation scripts
│   ├── build_graph.py              # Build Neo4j knowledge graph from raw text
│   ├── evaluate_pipeline.py        # Run evaluation on benchmark queries
│   ├── index_documents.py          # Vectorize & upload corpus to Qdrant
│   └── test_redis_cache.py         # Redis semantic cache verification test
├── app.py                          # FastAPI application & REST/SSE endpoints
├── docker-compose.yml              # Complete 4-tier stack container orchestration
├── Dockerfile                      # Production multi-stage Python 3.13 container
├── pyproject.toml                  # Python package specifications (uv)
└── README.md                       # System documentation
```

---

## 🚀 Quickstart Guide

### Prerequisites
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) (with WSL2 enabled on Windows)
* OpenAI API Key (`OPENAI_API_KEY`)

### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/BaoNguyenz/End-to-end-Medical-Ai-Chatbox.git
cd End-to-end-Medical-Ai-Chatbox

# Create environment configuration
cp .env.example .env
# Edit .env and enter your OPENAI_API_KEY
```

### 2. Launch with Docker Compose (Recommended)
```bash
# Start all 4 services: FastAPI Web, Redis Stack, Qdrant, Neo4j
docker compose up -d --build
```

Access the interfaces:
* 🌐 **Web UI Application**: [http://localhost:8080](http://localhost:8080)
* 📊 **FastAPI Interactive Docs**: [http://localhost:8080/docs](http://localhost:8080/docs)
* 🔴 **Qdrant Vector Dashboard**: [http://localhost:6335/dashboard](http://localhost:6335/dashboard)
* 🕸️ **Neo4j Graph Browser**: [http://localhost:7475](http://localhost:7475) *(User: `neo4j` / Password: `password123`)*

---

## 💻 Local Development & CLI

To run the system outside Docker for development or testing:

```bash
# 1. Install dependencies using uv
pip install uv
uv sync

# 2. Build React Frontend bundle
npm --prefix frontend install
npm --prefix frontend run build

# 3. Start interactive CLI
uv run python main.py

# 4. Or start FastAPI dev server
uv run python app.py
```

### Interactive CLI Commands
```text
User> /mode hybrid       # Change retrieval strategy (auto | hybrid | vector | bm25 | graph)
User> /graph on          # Toggle Neo4j graph traversal
User> /cache             # View Redis cache hit rate & statistics
User> /eval              # Run benchmark evaluation suite
User> /help              # Show help menu
User> /quit              # Exit session
```

---

## 📊 API Reference

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/query/stream` | `POST` | **Primary clinical query endpoint** with Server-Sent Events (SSE) token streaming. |
| `/api/query` | `POST` | Standard synchronous query endpoint returning full JSON payload. |
| `/api/health` | `GET` | Healthcheck returning status of Qdrant, Neo4j, BM25, and Redis. |
| `/api/stats` | `GET` | Corpus metrics (vector counts, graph nodes, relationships). |
| `/api/cache/stats` | `GET` | Redis Semantic Cache telemetry (hits, misses, hit rate %). |
| `/api/cache/clear` | `DELETE` | Flush all cached queries from Redis memory. |

---

## ⚖️ License & Disclaimer

* **Medical Disclaimer**: GaleMed AI is designed strictly for **educational and clinical reference purposes** based on *The Gale Encyclopedia of Medicine (3rd Edition)*. It does not provide formal medical diagnoses or replace professional medical consultations.
* **License**: MIT Open Source License.
