# GaleMed AI — Clinical GraphRAG & Hybrid Intelligence System

<p align="left">
  <a href="#-evaluation-metrics--performance">
    <img src="https://img.shields.io/badge/Answer_Faithfulness-88.5%25-brightgreen" alt="Answer Faithfulness">
  </a>
  <a href="#-evaluation-metrics--performance">
    <img src="https://img.shields.io/badge/Semantic_Cache_HIT-<10ms-orange" alt="Cache Latency">
  </a>
  <a href="#-evaluation-metrics--performance">
    <img src="https://img.shields.io/badge/Avg_RAG_Latency-4.25s-blue" alt="Average Latency">
  </a>
</p>

<p align="left">
  <a href="#-technology-stack">
    <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white" alt="Python">
  </a>
  <a href="#-technology-stack">
    <img src="https://img.shields.io/badge/FastAPI-0.141-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  </a>
  <a href="#-technology-stack">
    <img src="https://img.shields.io/badge/React-19_Vite-61DAFB?logo=react&logoColor=black" alt="React">
  </a>
  <a href="#-technology-stack">
    <img src="https://img.shields.io/badge/Qdrant-Vector_DB-red" alt="Qdrant">
  </a>
  <a href="#-technology-stack">
    <img src="https://img.shields.io/badge/Neo4j-GraphRAG-008CC1?logo=neo4j&logoColor=white" alt="Neo4j">
  </a>
  <a href="#-technology-stack">
    <img src="https://img.shields.io/badge/Redis_Stack-Semantic_Cache-DC382D?logo=redis&logoColor=white" alt="Redis">
  </a>
  <a href="#-technology-stack">
    <img src="https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?logo=openai&logoColor=white" alt="OpenAI">
  </a>
  <a href="#-getting-started">
    <img src="https://img.shields.io/badge/Docker-Supported-blue?logo=docker&logoColor=white" alt="Docker">
  </a>
</p>

An advanced, production-grade Clinical Retrieval-Augmented Generation (RAG) system designed to solve complex medical information synthesis grounded strictly in **The Gale Encyclopedia of Medicine (3rd Edition)**. The system features semantic chunking, multi-stage hybrid search (Dense Vector + BM25 with RRF), clinical knowledge graph traversal (Neo4j GraphRAG), query transformation (HyDE & Decomposition), post-retrieval reranking (Cross-Encoder & MMR), sub-10ms Redis HNSW Semantic Caching, and real-time Server-Sent Events (SSE) token streaming.

---

## 🏗️ Architecture Overview

The system operates on an End-to-End pipeline combining Vector Search, Keyword Search, Knowledge Graph Traversal, and Semantic In-Memory Caching:

```mermaid
flowchart TD
    %% Theme Styling matching main branch
    classDef storage fill:#e3f2fd,stroke:#1e88e5,stroke-width:2px,color:#0d47a1;
    classDef action fill:#e8f5e9,stroke:#43a047,stroke-width:2px,color:#1b5e20;
    classDef routing fill:#fffde7,stroke:#fbc02d,stroke-width:2px,color:#f57f17;
    classDef input fill:#ffe0b2,stroke:#fb8c00,stroke-width:2px,color:#e65100;
    classDef output fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px,color:#4a148c;
    classDef cache fill:#ffebee,stroke:#e53935,stroke-width:2px,color:#b71c1c;

    subgraph INGESTION ["OFFLINE DATA INGESTION PIPELINE"]
        Docs[Gale Medical Encyclopedia] --> Chunk[Semantic Section Chunking]
        
        %% Vector & BM25
        Chunk --> Embed[Text Embeddings<br/>all-MiniLM-L6-v2] --> VecDB[(Qdrant Vector DB<br/>13,350 Chunks)]
        Chunk --> BuildBM25[BM25 Indexing] --> BM25DB[(BM25 Lexical Index)]
        
        %% GraphRAG
        Chunk --> GraphExtract[LLM Clinical Entity-Relation Extract] --> GraphDB[(Neo4j Graph DB<br/>288 Diseases · 333 Meds · 508 Symptoms)]
    end

    subgraph CACHING ["ACCELERATION & SAFETY LAYER"]
        Q[Patient / Clinician Query] --> Safety{Emergency Guardrail}
        Safety -- "Critical / Anaphylaxis" --> EmergAns([Immediate 911 Triage])
        Safety -- "Standard Query" --> RedisCache[(Redis Semantic Cache<br/>HNSW Vector Index)]
        RedisCache -- "Similarity >= 92%" --> CacheHit([⚡ Instant Cache HIT<br/>&lt; 10ms Token Stream])
    end

    subgraph RETRIEVAL ["ONLINE RETRIEVAL PIPELINE (Real-Time)"]
        RedisCache -- "Cache MISS" --> Router{Query Router}
        
        %% Vector / Hybrid
        Router -- "Hybrid Search" --> QTrans[Query Transformation]
        QTrans --> HyDE[HyDE - Hypothetical Document]
        QTrans --> Decompose[Decomposer - Subqueries]
        
        HyDE & Decompose --> SearchEngine(Hybrid Search Engine)
        SearchEngine --> |Vector Search| VecDB
        SearchEngine --> |Keyword Search| BM25DB
        
        %% Merging & Reranking
        VecDB & BM25DB --> RRF[RRF Fusion]
        RRF --> |Top 50 Chunks| Merge[Merge Candidates]

        %% GraphRAG
        Router -- "GraphRAG" --> GraphSearch[GraphRAG Search]
        GraphSearch --> |Entity & Cypher Traversal| GraphDB
        GraphDB --> |Top 5 Results| Merge

        %% Post-Retrieval Pipeline
        Merge --> |Candidate Chunks| Rerank[Cross-Encoder Reranker<br/>ms-marco-MiniLM-L-6-v2]
        Rerank --> |Top 20 Chunks| MMR[MMR Diversity Filter<br/>lambda=0.7]
        MMR --> |Top 10 Chunks| Context[Verified Medical Context]
    end

    subgraph GENERATION ["GENERATION & STREAMING PIPELINE"]
        Context --> Assemble[Medical Prompt Assembly<br/>+ Disclaimers]
        Assemble --> LLM[GPT-4o-mini Generator<br/>stream=True]
        LLM --> StreamToken([SSE Real-Time Stream<br/>to React UI])
        LLM -.-> StoreCache[Store Query & Answer<br/>in Redis Cache] -.-> RedisCache
    end

    class VecDB,BM25DB,GraphDB storage;
    class RedisCache,CacheHit,StoreCache cache;
    class Chunk,Embed,BuildBM25,GraphExtract,RRF,Rerank,MMR,GraphSearch,Assemble,Merge action;
    class Router,QTrans,HyDE,Decompose,Safety routing;
    class Q input;
    class LLM,EmergAns,StreamToken output;
```

---

## 🛠️ Technology Stack

| Layer | Technologies / Frameworks | Purpose |
|---|---|---|
| **Vector Database** | Qdrant (`qdrant/qdrant:latest`) | High-performance vector similarity search storing 13,350 chunk embeddings. |
| **Graph Database** | Neo4j (`neo4j:5-community`) | Medical Knowledge Graph capturing multi-hop relationships (Diseases, Medications, Symptoms, Procedures). |
| **Semantic Cache** | Redis Stack (`redis/redis-stack-server`) | RediSearch in-memory HNSW vector index for sub-10ms query deduplication. |
| **Keyword Search** | BM25 (`rank-bm25`) | Classical lexical search for exact medical terminologies, drug brands, and anatomical terms. |
| **Embeddings & Reranking** | SentenceTransformers (`all-MiniLM-L6-v2`, `cross-encoder/ms-marco-MiniLM-L-6-v2`) | Local dense embedding generation and Cross-Encoder passage reranking. |
| **Orchestration / LLM** | OpenAI API (`gpt-4o-mini`), LangChain | Clinical intent classification, query decomposition, HyDE, and real-time streaming generation. |
| **Backend Framework** | FastAPI, Uvicorn (Python 3.13) | Production-ready asynchronous API server with Server-Sent Events (SSE) streaming endpoints. |
| **Frontend UI** | React 19, Vite, Vanilla CSS | Modern clinical dashboard with real-time token streaming, cache telemetry, and source inspection. |

---

## 🌟 Key Features

*   **Semantic Section Chunking:** Splits encyclopedia entries by medical structure (Definition, Causes, Symptoms, Treatments) rather than arbitrary token boundaries, preserving clinical integrity.
*   **Hybrid Search & RRF:** Merges dense vector representations (Qdrant) and lexical keyword matching (BM25) using **Reciprocal Rank Fusion (RRF)** to retrieve both contextual concepts and specific medical terms.
*   **GraphRAG (Neo4j):** Extracts entities and explicit clinical relationships (`TREATS`, `CAUSES`, `CONTRAINDICATED_WITH`, `SYMPTOM_OF`) to answer complex relational and multi-hop queries that standard vector search misses.
*   **Sub-10ms Redis Semantic Caching:** Checks incoming questions against a 384-dimensional HNSW index in Redis. Cache hits return answers from RAM in **< 10ms**, cutting OpenAI API costs to **$0**.
*   **Query Transformation:**
    *   **HyDE (Hypothetical Document Embeddings):** Generates hypothetical medical answers to expand short clinical inquiries.
    *   **Decomposition:** Breaks down complex, multi-symptom inquiries into discrete sub-queries.
*   **Post-Retrieval Pipeline:**
    *   **Cross-Encoder Reranker:** Scores exact query-chunk pairs with cross-attention to resolve dense-retrieval ranking noise.
    *   **MMR (Maximal Marginal Relevance):** Filters redundant chunks and enforces clinical context diversity.
*   **Real-Time Token Streaming (SSE):** Delivers instantaneous time-to-first-token (TTFT ~250ms) via FastAPI `text/event-stream`.
*   **Clinical Safety Reflex Engine:** Identifies life-threatening emergencies (e.g. crushing chest pain, anaphylaxis) for immediate triage guidance and always attaches educational medical disclaimers.

---

## 📁 Project Structure

```text
├── Data/
│   └── gale_encyclopedia_data/ # Source encyclopedia Markdown files
├── cache/                      # Local BM25 persisted corpus & index files
├── src/                        # Core codebase
│   ├── config.py               # Settings & configuration (Pydantic)
│   ├── models.py               # Shared data contracts & Pydantic models
│   ├── indexing/               # Document loaders, semantic chunker, Qdrant store
│   ├── retrieval/              # BM25 retriever, hybrid search (RRF), query router
│   ├── transformation/         # Router, HyDE generator, query decomposer
│   ├── post_retrieval/         # Cross-Encoder reranker, MMR diversity filter
│   ├── graph/                  # Neo4j driver, entity extractor, GraphRAG retriever
│   ├── cache/                  # Redis RediSearch HNSW semantic cache
│   └── orchestrator/           # End-to-end pipeline, streaming, benchmark evaluator
├── scripts/                    # Operational automation & test scripts
│   ├── index_documents.py      # Build Qdrant vector & BM25 indices
│   ├── build_graph.py          # Extract medical entities & build Neo4j graph
│   ├── test_redis_cache.py     # Verify Redis semantic cache HIT/MISS latency
│   └── evaluate_pipeline.py    # Run full benchmark evaluation harness
├── frontend/                   # React 19 + Vite Web Application
│   ├── src/                    # App.jsx, index.css, main.jsx
│   └── dist/                   # Production compiled assets
├── app.py                      # FastAPI Backend API & SSE streaming entrypoint
├── main.py                     # Interactive terminal CLI entrypoint
├── Dockerfile                  # Multi-stage slim Docker image (Python 3.13)
├── docker-compose.yml          # Orchestrates Web, Qdrant, Neo4j, and Redis Stack
├── .dockerignore               # Build optimization exclusions
└── pyproject.toml              # Project metadata & dependencies managed via uv
```

---

## 🛠️ Getting Started

### 1. Common Pre-requisite: Environment Setup

Before selecting a method below, clone the repository, copy `.env.example` to `.env` and fill in your OpenAI API Key:

```bash
# Clone the repository and navigate inside
git clone https://github.com/BaoNguyenz/End-to-end-Medical-Ai-Chatbox.git
cd End-to-end-Medical-Ai-Chatbox

# Copy environmental file
cp .env.example .env
```
Fill in your `OPENAI_API_KEY` in `.env`. Ensure other settings are left to defaults for standard setup.

---

### 🐳 Method 1: Docker Compose Deployment (Recommended)
This runs the entire 4-service stack inside lightweight Docker containers without needing Python or local packages.

#### 1. Start all containers (Databases + Redis Cache + Web Application)
```bash
docker compose up -d --build
```
Docker will pull Qdrant, Neo4j, Redis Stack Server, install dependencies using `uv` inside a multi-stage Python 3.13 container, pre-download NLTK tokenizers, and start the FastAPI server on port `8080`.

#### 2. Seed and Index documents (If databases are newly initialized)
```bash
# A. Build hybrid search index (BM25 + Qdrant vectors)
docker compose exec web python scripts/index_documents.py

# B. Build GraphRAG Knowledge Graph in Neo4j
docker compose exec web python scripts/build_graph.py
```

#### 3. Access Services
- **Web UI Application:** `http://localhost:8080` (Interactive clinical chat interface)
- **FastAPI Interactive Docs:** `http://localhost:8080/docs`
- **Qdrant DB Dashboard:** `http://localhost:6335/dashboard`
- **Neo4j DB Browser:** `http://localhost:7475` *(Credentials: `neo4j` / `password123`)*
- **Redis Stack Server:** `localhost:6379`

To stop all services: `docker compose down`

---

### 💻 Method 2: Local Development (Best for Editing Code)
This method runs databases in Docker containers but executes Python scripts and the FastAPI web server directly on your host machine.

#### 1. Setup local environment using `uv`
Ensure you have [Python 3.13+](https://www.python.org/downloads/) and [uv](https://github.com/astral-sh/uv) installed.
```bash
# Create local virtual environment and install dependencies
uv sync
```

#### 2. Spin up only the Databases & Cache
```bash
# Starts Qdrant, Neo4j, and Redis Stack containers
docker compose up -d qdrant neo4j redis
```

#### 3. Build indices & Build Frontend
```bash
# Load documents to Qdrant & build local BM25 index
uv run python scripts/index_documents.py

# Extract graph entities & push to Neo4j
uv run python scripts/build_graph.py

# Build React Frontend
npm --prefix frontend install
npm --prefix frontend run build
```

#### 4. Run Interactive Interfaces
- **Terminal CLI Chat:**
  ```bash
  uv run python main.py
  ```
  *(Type `/help` to see options, `/mode <mode>` to switch algorithms, `/cache` for telemetry, `/quit` to exit)*

- **Backend API Server (with auto-reload on changes):**
  ```bash
  uv run uvicorn app:app --reload --port 8000
  ```
  Open `http://localhost:8000` to interact with the GUI, or visit `http://localhost:8000/docs` to test endpoints.

---

## 📊 Evaluation Metrics & Performance

The RAG pipeline is evaluated end-to-end using Ragas-aligned metrics (Context Relevance & Answer Faithfulness) evaluated via LLM-as-a-judge across **The Gale Encyclopedia of Medicine** benchmark dataset.

### Summary Metrics

| Metric | Score / Value | Description |
|---|---|---|
| **Answer Faithfulness** | **88.5%** | Measures whether all claims in the generated clinical response are strictly grounded in retrieved encyclopedia context (zero hallucinations). |
| **Context Relevance** | **0.4620** | Assesses precision of retrieved chunks after Cross-Encoder reranking and MMR filtering relative to clinical query intent. |
| **Cache HIT Latency** | **&lt; 10ms** | Response latency when semantically matching queries are retrieved directly from Redis RAM. |
| **Average RAG Latency** | **4.25s** | Total end-to-end round trip time on Cache MISS (including multi-stage retrieval and LLM streaming). |

### Average Latency Breakdown per Stage (Cache MISS)
*   **Query Classification (Router):** `0.001s` (Deterministic intent mapping)
*   **Vector/Keyword Retrieval:** `0.840s` (Qdrant & BM25 search)
*   **GraphRAG Traversal (Neo4j):** `1.120s` (Entity linking & Cypher traversal)
*   **Post-Retrieval Processing:** `0.450s` (Cross-Encoder reranking & MMR)
*   **LLM Streaming Generation:** `1.840s` (First token ~250ms via SSE)

---

## 🧪 Testing & Verification

The project includes test scripts for verifying each architectural component:

```bash
# 1. Verify Vector + BM25 Hybrid Search
uv run python scripts/test_hybrid_search.py

# 2. Verify Query Transformation (Routing, HyDE, Decomposition)
uv run python scripts/test_query_transformation.py

# 3. Verify Cross-Encoder Reranking & MMR Diversity
uv run python scripts/test_post_retrieval.py

# 4. Verify Neo4j Graph Database & GraphRAG queries
uv run python scripts/test_graph.py

# 5. Verify Redis HNSW Semantic Cache (MISS vs HIT latency)
uv run python scripts/test_redis_cache.py

# 6. Run complete evaluation benchmark harness
uv run python scripts/evaluate_pipeline.py
```

---

## ⚖️ Medical Disclaimer

*GaleMed AI is strictly an educational and clinical research demonstration tool powered by The Gale Encyclopedia of Medicine (3rd Edition). It does not provide medical diagnoses, clinical advice, or treatment plans. In any real-world health emergency, always consult a licensed medical doctor or contact emergency services (911).*
