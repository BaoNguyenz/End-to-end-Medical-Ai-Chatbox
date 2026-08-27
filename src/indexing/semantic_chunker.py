"""
semantic_chunker.py
Split documents into semantically coherent chunks based on
sentence-level embedding similarity.
"""

from __future__ import annotations

import re
import uuid
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.models import Document, Chunk


class SemanticChunker:
    """
    Splits documents into chunks where consecutive sentences share
    high semantic similarity.  Falls back to paragraph splitting for
    very short documents.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.75,
        max_chunk_size: int = 1000,
        min_chunk_size: int = 100,
        _model: Optional[SentenceTransformer] = None,
    ) -> None:
        self.model = _model or SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    # ------------------------------------------------------------------
    # Sentence splitting
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """
        Split text into sentence-level units, keeping code blocks
        and tables intact as single units.
        """
        # 1. Protect code blocks (``` ... ```)
        code_blocks: list[str] = []
        def _save_code(m: re.Match) -> str:
            code_blocks.append(m.group(0))
            return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

        text = re.sub(r"```[\s\S]*?```", _save_code, text)

        # 2. Protect tables (lines starting with |)
        table_blocks: list[str] = []
        def _save_table(m: re.Match) -> str:
            table_blocks.append(m.group(0))
            return f"__TABLE_BLOCK_{len(table_blocks) - 1}__"

        text = re.sub(r"(?:^\|.+\|\s*\n)+", _save_table, text, flags=re.MULTILINE)

        # 3. Split by double-newline (paragraph) first, then by sentence
        paragraphs = re.split(r"\n{2,}", text)

        sentences: list[str] = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            # If paragraph is a placeholder, keep as-is
            if para.startswith("__CODE_BLOCK_") or para.startswith("__TABLE_BLOCK_"):
                sentences.append(para)
                continue
            # Split by sentence-ending punctuation
            parts = re.split(r"(?<=[.!?])\s+", para)
            sentences.extend(p.strip() for p in parts if p.strip())

        # 4. Restore code blocks and tables
        restored: list[str] = []
        for s in sentences:
            for i, cb in enumerate(code_blocks):
                s = s.replace(f"__CODE_BLOCK_{i}__", cb)
            for i, tb in enumerate(table_blocks):
                s = s.replace(f"__TABLE_BLOCK_{i}__", tb)
            restored.append(s)

        return restored

    # ------------------------------------------------------------------
    # Similarity computation
    # ------------------------------------------------------------------

    def _compute_similarities(self, sentences: list[str]) -> list[float]:
        """Compute cosine similarities between consecutive sentence embeddings."""
        if len(sentences) < 2:
            return []

        embeddings = self.model.encode(sentences, normalize_embeddings=True)
        similarities: list[float] = []

        for i in range(len(embeddings) - 1):
            sim = float(np.dot(embeddings[i], embeddings[i + 1]))
            similarities.append(sim)

        return similarities

    # ------------------------------------------------------------------
    # Breakpoint detection
    # ------------------------------------------------------------------

    def _find_breakpoints(self, similarities: list[float]) -> list[int]:
        """
        Return indices where similarity drops below threshold.
        These become chunk boundaries.
        """
        return [
            i + 1
            for i, sim in enumerate(similarities)
            if sim < self.similarity_threshold
        ]

    # ------------------------------------------------------------------
    # Chunk merging / splitting helpers
    # ------------------------------------------------------------------

    def _merge_small_chunks(self, chunks: list[str]) -> list[str]:
        """Merge chunks smaller than min_chunk_size with the previous chunk."""
        if not chunks:
            return chunks

        merged: list[str] = [chunks[0]]
        for chunk in chunks[1:]:
            if len(merged[-1]) < self.min_chunk_size:
                merged[-1] = merged[-1] + "\n\n" + chunk
            else:
                merged.append(chunk)

        # Handle last chunk being too small
        if len(merged) > 1 and len(merged[-1]) < self.min_chunk_size:
            merged[-2] = merged[-2] + "\n\n" + merged[-1]
            merged.pop()

        return merged

    def _split_large_chunk(self, text: str) -> list[str]:
        """Force-split a chunk that exceeds max_chunk_size."""
        if len(text) <= self.max_chunk_size:
            return [text]

        parts: list[str] = []
        sentences = re.split(r"(?<=[.!?\n])\s+", text)
        current = ""

        for sentence in sentences:
            if len(current) + len(sentence) + 1 > self.max_chunk_size and current:
                parts.append(current.strip())
                current = sentence
            else:
                current = current + " " + sentence if current else sentence

        if current.strip():
            parts.append(current.strip())

        return parts if parts else [text]

    # ------------------------------------------------------------------
    # Main chunking method
    # ------------------------------------------------------------------

    def chunk_document(self, doc: Document) -> list[Chunk]:
        """
        Split a document into semantically coherent chunks.

        Returns:
            List of Chunk objects with metadata.
        """
        text = doc.page_content.strip()

        # Edge case: very short document
        if len(text) <= self.min_chunk_size:
            return [Chunk(
                content=text,
                chunk_id=f"{doc.doc_id}_chunk_0",
                doc_id=doc.doc_id,
                document_type=doc.document_type,
                chunk_index=0,
                metadata={**doc.metadata},
            )]

        # Step 1: Split into sentences
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [Chunk(
                content=text,
                chunk_id=f"{doc.doc_id}_chunk_0",
                doc_id=doc.doc_id,
                document_type=doc.document_type,
                chunk_index=0,
                metadata={**doc.metadata},
            )]

        # Step 2: Compute similarities and find breakpoints
        similarities = self._compute_similarities(sentences)
        breakpoints = self._find_breakpoints(similarities)

        # Step 3: Create raw chunks from breakpoints
        raw_chunks: list[str] = []
        prev = 0
        for bp in breakpoints:
            chunk_text = "\n".join(sentences[prev:bp]).strip()
            if chunk_text:
                raw_chunks.append(chunk_text)
            prev = bp
        # Last chunk
        last_chunk = "\n".join(sentences[prev:]).strip()
        if last_chunk:
            raw_chunks.append(last_chunk)

        # Step 4: Merge small chunks
        raw_chunks = self._merge_small_chunks(raw_chunks)

        # Step 5: Split oversized chunks
        final_chunks: list[str] = []
        for chunk in raw_chunks:
            final_chunks.extend(self._split_large_chunk(chunk))

        # Step 6: Create Chunk objects
        chunks: list[Chunk] = []
        for i, content in enumerate(final_chunks):
            chunks.append(Chunk(
                content=content,
                chunk_id=f"{doc.doc_id}_chunk_{i}",
                doc_id=doc.doc_id,
                document_type=doc.document_type,
                chunk_index=i,
                metadata={**doc.metadata},
            ))

        return chunks

    def chunk_documents(self, docs: list[Document]) -> list[Chunk]:
        """Chunk multiple documents. Returns flat list of all chunks."""
        all_chunks: list[Chunk] = []
        for doc in docs:
            doc_chunks = self.chunk_document(doc)
            all_chunks.extend(doc_chunks)
            print(
                f"  [{doc.doc_id}] {len(doc_chunks)} chunks "
                f"(avg {sum(len(c.content) for c in doc_chunks) // max(len(doc_chunks), 1)} chars)"
            )
        print(f"\n[Chunker] Total: {len(all_chunks)} chunks from {len(docs)} documents")
        return all_chunks
