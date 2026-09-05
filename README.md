# GaleMed AI — Enterprise Clinical RAG & Knowledge Graph System

<p align="left">
  <a href="#-automated-cicd--cloud-deployment">
    <img src="https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?logo=githubactions&logoColor=white" alt="CI/CD">
  </a>
  <a href="#-automated-cicd--cloud-deployment">
    <img src="https://img.shields.io/badge/Cloud-Microsoft_Azure_VM-0078D4?logo=microsoftazure&logoColor=white" alt="Azure">
  </a>
  <a href="#-technology-stack">
    <img src="https://img.shields.io/badge/Python-3.13%2B-blue?logo=python&logoColor=white" alt="Python">
  </a>
  <a href="#-technology-stack">
    <img src="https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  </a>
  <a href="#-technology-stack">
    <img src="https://img.shields.io/badge/Qdrant-13%2C350_Vectors-red?logo=qdrant&logoColor=white" alt="Qdrant">
  </a>
  <a href="#-technology-stack">
    <img src="https://img.shields.io/badge/Neo4j-GraphRAG_5.0-008CC1?logo=neo4j&logoColor=white" alt="Neo4j">
  </a>
  <a href="#-technology-stack">
    <img src="https://img.shields.io/badge/Redis-Semantic_Cache-DC382D?logo=redis&logoColor=white" alt="Redis">
  </a>
  <a href="#-technology-stack">
    <img src="https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?logo=openai&logoColor=white" alt="OpenAI">
  </a>
  <a href="#-getting-started">
    <img src="https://img.shields.io/badge/Docker-Multi--stage_Build-2496ED?logo=docker&logoColor=white" alt="Docker">
  </a>
</p>

**GaleMed AI** is an advanced, production-grade clinical Decision-Support & Retrieval-Augmented Generation (RAG) system. Designed to navigate vast medical encyclopedias and clinical knowledge bases, the platform combines **Dense Vector Retrieval (Qdrant)**, **Lexical Keyword Search (BM25)**, **Knowledge Graph Traversal (Neo4j GraphRAG)**, **L1 Semantic Caching (Redis Stack)**, and **Precision Cross-Encoder Reranking** to deliver zero-hallucination, evidence-backed medical insights with sub-second latency.

---

## 🏗️ Architecture Overview

The system operates on an authentic **Dual-Pipeline Architecture** separating **Offline Medical Knowledge Ingestion** from **Online Real-Time Clinical Retrieval & Generation**:

<p align="center">
  <img src="docs/architecture.svg" alt="GaleMed AI Dual-Pipeline System Architecture" width="100%" />
</p>
---

## 🌟 Key Features

*   **⚡ L1 Redis Semantic Cache:** Cosine similarity ($\ge 0.92$) • Sub-10ms latency • Zero-cost query reuse.
*   **🧬 Neo4j GraphRAG:** Multi-hop reasoning • 1,465 entities (Disease, Drug, Symptom) • 441 relationships.
*   **🔍 Hybrid Search & RRF:** Dense Vector (Qdrant HNSW) + Sparse Lexical (BM25) • Reciprocal Rank Fusion ($k=60$).
*   **🎯 Cross-Encoder Reranker:** Precision passage scoring (`ms-marco-MiniLM-L-6-v2`) • MMR diversity filter.
*   **🔄 Query Transformation:** HyDE (Hypothetical Embeddings) • Multi-query Decomposition • Intent Routing.
*   **📊 Clinical Grounding:** Verifiable document IDs • Section-level citations • Strict zero-hallucination prompt.
---

## 📊 Dataset & Knowledge Graph Statistics

The knowledge base is constructed from verified medical encyclopedias and clinical reference literature:

<div align="center">

| Metric / Category | Volume | Details |
| :--- | :---: | :--- |
| **Medical Reference Documents** | **292** entries | Disease (107), Drug (54), General (84), Procedure (34), Test (13) |
| **Indexed Vector Chunks** | **13,350** chunks | Semantic-chunked (avg. 164 chars) with normalized embeddings |
| **Vector Store Collection** | medical_docs | Qdrant HNSW Index (M=16, ef_construct=100)
| **Graph Entities (Nodes)** | **1,465** nodes | Disease (288), Medication (333), Symptom (508), Procedure (99), Entry (237) |
| **Graph Relationships (Edges)** | **441** relations | `TREATS`, `HAS_SYMPTOM`, `REQUIRES_PROCEDURE`, `BELONGS_TO` |

</div>

---

## ⚡ Performance & Latency Benchmark

Benchmarked across clinical question sets comparing cache hits, hybrid retrieval, and full GraphRAG pipelines:

| Execution Mode | Average Latency | Context Relevance | Answer Faithfulness | Cost per Query |
| :--- | :---: | :---: | :---: | :---: |
| **Redis Semantic Cache Hit** | **~8 ms** ⚡ | 100% (Pre-verified) | 100% | **$0.0000** |
| **Dense Vector Only (Qdrant)** | **~190 ms** | 0.3810 | 79.2% | Standard |
| **Hybrid Search (Qdrant + BM25)** | **~240 ms** | 0.4130 | 83.3% | Standard |
| **Full GraphRAG + Cross-Encoder** | **~450 ms** | **0.4850** | **91.7%** | Standard |
| **End-to-End with GPT-4o-mini** | **~1.85 s** | — | — | ~$0.0003 |

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Core AI & Models** | OpenAI `gpt-4o-mini`, SentenceTransformers `all-MiniLM-L6-v2` | Intent reasoning, query transformation, sentence embeddings, and generation |
| **Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Post-retrieval cross-attention passage reranking |
| **Vector Database** | **Qdrant** | High-performance vector storage with HNSW index |
| **Graph Database** | **Neo4j 5** (APOC enabled) | Multi-hop clinical entity and relation traversal |
| **Semantic Cache** | **Redis Stack** | Sub-10ms vector similarity response cache |
| **Lexical Search** | **Rank-BM25** | In-memory exact keyword matching |
| **Backend API** | **FastAPI**, **Uvicorn** | Asynchronous HTTP API with streaming responses and OpenAPI docs |
| **Frontend UI** | **React**, **Vite**, Modern Glassmorphic CSS | Interactive clinical chat UI, live latency metrics, and citation preview |
| **Containerization** | **Docker**, **Docker Compose** | Multi-stage slim runtime build with non-root security |
| **CI/CD & Cloud** | **GitHub Actions**, **Microsoft Azure VM** | Automated syntax/lint validation and automated SSH cloud deployment |

---

## 🔄 Automated CI/CD & Cloud Deployment

The repository includes a production **Continuous Integration / Continuous Deployment (CI/CD)** pipeline powered by GitHub Actions and Microsoft Azure:

```mermaid
flowchart LR
    Dev[💻 Developer Push\nbranch: main] --> GHA[⚙️ GitHub Actions Runner]
    
    subgraph CI [1. Continuous Integration]
        GHA --> Setup[Setup Python 3.13]
        Setup --> Lint[Ruff Critical Syntax & Linter Check\n--select=E9,F63,F7,F82]
    end
    
    subgraph CD [2. Continuous Deployment]
        Lint --> SSH[SSH Key Handshake\nAzure Linux VM]
        SSH --> Pull[git pull origin main]
        Pull --> Build[docker compose up -d --build web]
        Build --> Clean[docker image prune -f]
        Clean --> Live[🌐 Live at http://98.70.58.126:8080]
    end
```

---

## 🚀 Quick Start Guide

### Prerequisites
- [Docker & Docker Compose](https://docs.docker.com/get-docker/) installed
- OpenAI API Key

### 1. Clone & Configure Environment
```bash
git clone https://github.com/BaoNguyenz/End-to-end-Medical-Ai-Chatbox.git
cd End-to-end-Medical-Ai-Chatbox

# Create environment file
cp .env.example .env
```
Open `.env` and set your `OPENAI_API_KEY`:
```env
OPENAI_API_KEY=sk-your-openai-api-key
```

---

### 2. Run with Docker Compose (Recommended)

#### Step A: Start all services
```bash
docker compose up -d
```
*Spins up `rag-web` (FastAPI), `rag-qdrant` (Vector DB), `rag-neo4j` (Graph DB), and `rag-redis` (Cache).*

#### Step B: Ingest Knowledge Base & Build Graph
```bash
# 1. Index 13,350 chunks into Qdrant & build BM25 corpus
docker compose run --rm web python scripts/index_documents.py

# 2. Populate Neo4j Knowledge Graph with medical entities
docker compose run --rm web python scripts/build_graph.py

# 3. Restart web server to bind freshly populated databases
docker compose restart web
```

#### Step C: Access Applications
- 🌐 **Web Chat Application:** `http://localhost:8080` (or `http://localhost:8000`)
- 📖 **Interactive API Documentation:** `http://localhost:8080/docs`
- 🩺 **System Health Endpoint:** `http://localhost:8080/api/health`
- 🗄️ **Qdrant Vector Dashboard:** `http://localhost:6335/dashboard`
- 🕸️ **Neo4j Graph Browser:** `http://localhost:7475` (Auth: `neo4j` / `password123`)

---

### 3. Local Development (Alternative)

```bash
# Install uv package manager
pip install uv

# Create virtual environment & install dependencies
uv venv
uv pip install -e ".[dev]"

# Start database containers only
docker compose up -d qdrant neo4j redis

# Ingest data & run locally
uv run python scripts/index_documents.py
uv run python scripts/build_graph.py
uv run uvicorn app:app --reload --port 8000
```

---

## 🧪 Testing & Verification

Run automated test suites to verify each layer of the pipeline independently:

```bash
# Test Hybrid Search (Vector + BM25 Fusion)
uv run python scripts/test_hybrid_search.py

# Test Query Transformations (HyDE & Decomposition)
uv run python scripts/test_query_transformation.py

# Test Cross-Encoder Reranking & MMR Diversity
uv run python scripts/test_post_retrieval.py

# Test Neo4j Knowledge Graph extraction & queries
uv run python scripts/test_graph.py
```

---

## 📄 License & Acknowledgments

This project is licensed under the MIT License. Developed as a comprehensive Enterprise Medical AI solution integrating state-of-the-art Hybrid Search, Knowledge Graph reasoning, and sub-second caching.
