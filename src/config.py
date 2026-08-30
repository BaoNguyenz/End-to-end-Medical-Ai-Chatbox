"""
config.py
Centralized configuration using Pydantic Settings.
Loads from .env file with type-safe defaults.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # --- Embedding ---
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384  # MiniLM output dimension

    # --- Cross-Encoder ---
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Qdrant ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "medical_docs"  # Changed from enterprise_docs

    # --- Neo4j ---
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password123"

    # --- HNSW Parameters ---
    hnsw_m: int = 32              # Max edges per node in HNSW graph
    hnsw_ef_construct: int = 200  # Search width during graph construction
    hnsw_ef_search: int = 100     # Search width during query

    # --- Chunking ---
    # Slightly lower threshold to better respect medical article boundaries
    chunk_similarity_threshold: float = 0.72
    # Smaller chunks for more precise medical fact retrieval
    chunk_max_size: int = 800
    chunk_min_size: int = 100

    # --- Search ---
    bm25_top_k: int = 50
    vector_top_k: int = 50
    hybrid_top_k: int = 20
    rerank_top_k: int = 10
    rrf_k: int = 60
    mmr_lambda: float = 0.5

    # --- Paths ---
    # Medical encyclopedia articles extracted as individual Markdown files
    data_dir: Path = ROOT_DIR / "Data" / "markdown_output"
    cache_dir: Path = ROOT_DIR / "cache"

    def ensure_dirs(self) -> None:
        """Create necessary directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


# Singleton instance
settings = Settings()
