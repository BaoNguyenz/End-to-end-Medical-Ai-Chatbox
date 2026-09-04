# GaleMed AI — Clinical GraphRAG & Hybrid Intelligence System

<p align="left">
  <a href="#-evaluation-metrics--performance">
    <img src="https://img.shields.io/badge/Negative_Rejection-92.4%25-brightgreen" alt="Negative Rejection">
  </a>
  <a href="#-evaluation-metrics--performance">
    <img src="https://img.shields.io/badge/Medical_Safety-84.8%25-brightgreen" alt="Medical Safety">
  </a>
  <a href="#-evaluation-metrics--performance">
    <img src="https://img.shields.io/badge/Answer_Relevance-76.2%25-green" alt="Answer Relevance">
  </a>
  <a href="#-evaluation-metrics--performance">
    <img src="https://img.shields.io/badge/Semantic_Cache_HIT-<10ms-orange" alt="Cache Latency">
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
    <img src="https://img.shields.io/badge/UI_Design-Stitch_MCP-6366F1?logo=google&logoColor=white" alt="Stitch MCP">
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

The system operates on an authentic **Dual-Pipeline Architecture** separating **Offline Medical Knowledge Ingestion** from **Online Real-Time Clinical Retrieval & Generation**:

<p align="center">
  <img src="docs/architecture.svg" alt="GaleMed AI Dual-Pipeline System Architecture" width="100%" />
</p>



## 🛠️ Technology Stack

| Layer | Technologies / Frameworks | Purpose |
|---|---|---|
| **Vector Database** | Qdrant (`qdrant:latest`) | Dense Vector Search, HNSW Indexing, Cosine Similarity |
| **Graph Database** | Neo4j (`neo4j:5-community`) | Medical Knowledge Graph, Multi-Hop GraphRAG, Cypher Queries |
| **Semantic Cache** | Redis Stack (`redis-stack-server`) | In-Memory Semantic Caching (<10ms), Vector Deduplication |
| **Keyword Search** | BM25 (`rank-bm25`) | Lexical Keyword Search, Exact Medical Term Matching, ICD Codes |
| **Embeddings & Reranking** | SentenceTransformers (`all-MiniLM-L6-v2`, `ms-marco-MiniLM-L-6-v2`) | 384-dim Dense Embeddings, Cross-Encoder Re-Ranking |
| **Orchestration / LLM** | OpenAI API (`gpt-4o-mini`), LangChain | Intent Routing, HyDE, Query Decomposition, Clinical Generation |
| **Backend Framework** | FastAPI, Uvicorn (Python 3.13) | Async REST API, SSE Token Streaming, Multi-Stage Docker |
| **Frontend UI** | React 19, Vite, Tailwind CSS | Clinical Chat UI, Real-Time SSE Stream, Cache Telemetry |
| **UI Design & Prototyping** | Google Stitch (MCP Protocol) | Design System Generation, Layout & Component Prototyping |

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

## 📚 Medical Knowledge Dataset & Data Ingestion Pipeline

The foundation of GaleMed AI is built upon an authoritative, multi-volume medical corpus that undergoes multi-stage transformation from raw publication files into dense vector embeddings, an inverted lexical index, and a structured clinical knowledge graph.

### 1. 📖 Source Corpus: The Gale Encyclopedia of Medicine (3rd Edition)

* **Origin & Scope:** A comprehensive, 5-volume clinical reference covering thousands of human medical conditions, diagnostic modalities, surgical procedures, and pharmacology monographs.
* **Standardized Medical Topic Structure:** Each entry in the encyclopedia follows a structured clinical hierarchy:
  * `### Definition`: Precise clinical scoping of the disease or topic.
  * `### Description`: Pathophysiology, epidemiology, and disease progression.
  * `### Causes and symptoms`: Etiology, risk factors, and hallmark clinical manifestations.
  * `### Diagnosis`: Physical exams, lab assays, imaging protocols, and differential diagnoses.
  * `### Treatment`: Standard-of-care pharmacotherapy, surgical interventions, and supportive care.
  * `### Prognosis & Prevention`: Expected patient outcomes, complications, and prophylactic measures.
  * `### Key terms`: Formal medical definitions, anatomical sites, and pharmacological classes.

---

### 2. ⚙️ End-to-End Data Ingestion Architecture

```mermaid
flowchart LR
    subgraph S1 ["1. PDF Extraction & Normalization"]
        PDF[Gale Medical PDF<br/>64.3 MB] --> PyMuPDF[PyMuPDF4LLM Engine]
        PyMuPDF --> Dehyphen[De-hyphenation & Cleaning<br/>cholesty-
ramine → cholestyramine]
        Dehyphen --> TopicSplit[Entry Classifier & Splitter<br/>Filter Standard Subheadings]
    end

    subgraph S2 ["2. Semantic Section Chunking"]
        TopicSplit --> SyntaxShield[Clinical Block Shielding]
        SyntaxShield --> SentSplit[Sentence Tokenization]
        SentSplit --> CosineSim[Consecutive Cosine Similarity<br/>all-MiniLM-L6-v2]
        CosineSim --> Breakpoint[Dynamic Breakpoint Detection<br/>similarity < 0.75]
        Breakpoint --> SizeBounds[Adaptive Length Bounds<br/>100 ≤ chars ≤ 1000]
    end

    subgraph S3 ["3. Triple-Store Indexing"]
        SizeBounds --> DenseEmbed[Dense Embedding<br/>384-dim Vector] --> Qdrant[(Qdrant Vector DB<br/>13,350 vectors)]
        SizeBounds --> BM25Token[Lexical Tokenizer<br/>Stopword & Case Normalization] --> BM25[(BM25 Retriever)]
        TopicSplit --> LLMExtract[LLM Entity-Relation Extractor<br/>GPT-4o-mini + SHA-256 Cache] --> Neo4j[(Neo4j Graph DB<br/>GraphRAG)]
    end
```

---

---

### 3. 📊 Data & Indexing Statistics Summary

| Metric / Artifact | Value / Parameter | Purpose |
|---|:---:|---|
| **Raw Source PDF** | `64.3 MB` | *The Gale Encyclopedia of Medicine (3rd Edition)* |
| **Extracted Entries** | `20+` Key Reference Modules | Standardized medical condition and pharmacological profiles |
| **Indexed Vector Chunks** | `13,350` Vectors | Total granular semantic passages stored in Qdrant |
| **Embedding Dimensions** | `384` Dimensions | `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector Similarity Metric** | `Cosine` (HNSW) | Fast high-dimensional semantic search ($M=16, 	ext{ef}=100$) |
| **Knowledge Graph Schema** | 5 Nodes, 8 Relations | `Disease`, `Medication`, `Symptom`, `Procedure`, `Entry` |
| **Extraction Cache** | SHA-256 (`cache/`) | Zero duplicate API overhead during graph regeneration |

---

## 📁 Project Structure

```text
├── Data/
│   ├── The-Gale-Encyclopedia-of-Medicine-3rd-Edition-staibabussalamsula.pdf  # Raw 64.3MB Source PDF
│   ├── convert_pdf_to_md.py          # PyMuPDF4LLM PDF-to-Markdown extractor
│   ├── markdown_output/              # Structured encyclopedia Markdown entries
│   └── medical_benchmark_100.json    # Clinical benchmark evaluation dataset
├── cache/                            # Persisted BM25 tokens, entity cache & embeddings
├── src/                              # Core codebase
│   ├── config.py                     # Settings & configuration (Pydantic)
│   ├── models.py                     # Shared data contracts & Pydantic models
│   ├── indexing/                     # Document loaders, semantic chunker, Qdrant store
│   ├── retrieval/                    # BM25 retriever, hybrid search (RRF), query router
│   ├── transformation/               # Router, HyDE generator, query decomposer
│   ├── post_retrieval/               # Cross-Encoder reranker, MMR diversity filter
│   ├── graph/                        # Neo4j driver, entity extractor, GraphRAG retriever
│   ├── cache/                        # Redis RediSearch HNSW semantic cache
│   └── orchestrator/                 # End-to-end pipeline, streaming, benchmark evaluator
├── scripts/                          # Operational automation & test scripts
│   ├── index_documents.py            # Build Qdrant vector & BM25 indices
│   ├── build_graph.py                # Extract medical entities & build Neo4j graph
│   ├── test_redis_cache.py           # Verify Redis semantic cache HIT/MISS latency
│   └── evaluate_pipeline.py          # Run full benchmark evaluation harness
├── frontend/                         # React 19 + Vite Web Application
│   ├── src/                          # App.jsx, index.css, main.jsx
│   └── dist/                         # Production compiled assets
├── app.py                            # FastAPI Backend API & SSE streaming entrypoint
├── main.py                           # Interactive terminal CLI entrypoint
├── Dockerfile                        # Multi-stage slim Docker image (Python 3.13)
├── docker-compose.yml                # Orchestrates Web, Qdrant, Neo4j, and Redis Stack
├── .dockerignore                     # Build optimization exclusions
└── pyproject.toml                    # Project metadata & dependencies managed via uv
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

The RAG pipeline is evaluated end-to-end using a comprehensive clinical benchmark suite (**105 test cases**) across 8 clinical domains: *Cardiovascular, Respiratory, Pharmacology, Neuro-Psychiatry, Surgery & GI, Emergency Triage, Out-of-Scope*, and *Adversarial Injections*.

### 📈 End-to-End Evaluation Summary (105 Queries)

| Layer | Metric | Average Score | Description |
|---|---|:---:|---|
| **Retrieval** | **Context Relevance** | **0.5078** | Assesses how relevant and focused the retrieved chunks (post-MMR and Cross-Encoder reranking) are to the medical query. |
| **Generator** | **Answer Faithfulness** | **0.6286** | Measures whether clinical claims in the generated response are strictly grounded in retrieved encyclopedia context. |
| **Generator** | **Answer Relevance** | **0.7619** | Evaluates how directly and completely the synthesized answer addresses the user's clinical question. |
| **Generator** | **Medical Safety** | **0.8476** | Ensures answers do not prescribe dangerous dosages, fail to triage emergencies, or give harmful medical advice. |
| **Security** | **Negative Rejection** | **0.9238** | Measures the system's ability to safely reject adversarial jailbreaks, non-medical inquiries, and hallucinations. |

---

### 🩺 Scores by Clinical Domain

| Clinical Category | Queries (N) | Context Relevance | Faithfulness | Answer Relevance | Medical Safety |
|:---|:---:|:---:|:---:|:---:|:---:|
| 🫀 **Cardiovascular** | 15 | `0.556` | `0.700` | `0.900` | `0.933` |
| 🫁 **Respiratory** | 15 | `0.600` | `0.733` | `0.867` | `0.933` |
| 💊 **Pharmacology** | 15 | `0.489` | `0.667` | `0.800` | `0.867` |
| 🧠 **Neuro-Psychiatry** | 15 | `0.533` | `0.600` | `0.767` | `0.867` |
| 🩺 **Surgery & GI** | 15 | `0.511` | `0.667` | `0.800` | `0.867` |
| 🚨 **Emergency Triage** | 10 | `0.567` | `0.600` | `0.800` | `0.900` |
| ⛔ **Out of Scope** | 10 | `0.400` | `0.500` | `0.500` | `0.700` |
| 🛡️ **Adversarial Safety** | 10 | `0.300` | `0.400` | `0.500` | `0.600` |

---

### ⏱️ Latency & Cache Performance Breakdown

| Pipeline Stage | Cold Latency (MISS) | Warm Latency (HIT) | Implementation Details |
|---|:---:|:---:|---|
| **⚡ Semantic Cache Lookup** | `8ms` | **`< 10ms`** | Redis HNSW Vector Search (`all-MiniLM-L6-v2`, Cosine $\ge 0.90$) |
| **Query Routing & Transformation** | `350ms` | *Skipped* | Deterministic classification + HyDE / Decompose |
| **Hybrid Retrieval (Vector + BM25)** | `120ms` | *Skipped* | Qdrant (HNSW Cosine) + Rank-BM25 with RRF Fusion |
| **GraphRAG Traversal (Neo4j)** | `280ms` | *Skipped* | Entity linking & multi-hop Cypher path queries |
| **Post-Retrieval (Rerank + MMR)** | `150ms` | *Skipped* | `ms-marco-MiniLM-L-6-v2` + MMR ($\lambda=0.7$) |
| **GPT-4o-mini Generation (SSE)** | `1,200ms` | *Skipped* | Server-Sent Events token stream (TTFT ~250ms) |
| **🎯 Total Round Trip Time** | **`2.10s`** | **`< 10ms`** | **99.5% latency reduction & 100% cost reduction on HIT** |

---

## 🧪 Testing & Verification

The repository includes dedicated verification test suites for each subsystem:

```bash
# 1. Test Dense Vector + BM25 Hybrid Search (no API key required)
uv run python scripts/test_hybrid_search.py

# 2. Test Query Classification, HyDE, and Query Decomposition
uv run python scripts/test_query_transformation.py

# 3. Test Cross-Encoder Reranker & MMR Context Diversity Filter
uv run python scripts/test_post_retrieval.py

# 4. Test Neo4j Knowledge Graph extraction & GraphRAG traversal
uv run python scripts/test_graph.py

# 5. Test Redis Semantic Cache (verify sub-10ms HIT response)
uv run python scripts/test_redis_cache.py

# 6. Run full 105-query evaluation benchmark suite
uv run python scripts/evaluate_pipeline.py
```
