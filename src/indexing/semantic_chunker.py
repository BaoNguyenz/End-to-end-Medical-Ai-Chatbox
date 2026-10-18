"""
semantic_chunker.py
Split documents into semantically coherent chunks based on
sentence-level embedding similarity.
"""

from __future__ import annotations

import re
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
        min_chunk_size: int = 500,
        overlap_sentences: int = 1,
        _model: Optional[SentenceTransformer] = None,
    ) -> None:
        self.model = _model or SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_sentences = max(0, overlap_sentences)

    # ------------------------------------------------------------------
    # Markdown structure
    # ------------------------------------------------------------------

    @staticmethod
    def _split_markdown_sections(text: str) -> tuple[str, list[tuple[list[str], str]]]:
        """Split Markdown without allowing chunks to cross heading boundaries."""
        title = ""
        heading_stack: list[tuple[int, str]] = []
        sections: list[tuple[list[str], str]] = []
        body_lines: list[str] = []

        def flush() -> None:
            body = "\n".join(body_lines).strip()
            if body:
                sections.append(([heading for _, heading in heading_stack], body))
            body_lines.clear()

        for line in text.splitlines():
            match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
            if not match:
                body_lines.append(line)
                continue

            flush()
            level = len(match.group(1))
            heading = match.group(2).strip()
            if level == 1:
                title = heading
                heading_stack = []
            else:
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, heading))

        flush()
        return title, sections

    @staticmethod
    def _context_prefix(title: str, section_path: list[str]) -> str:
        labels = [label for label in [title, *section_path] if label]
        return " — ".join(labels)

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

    def _add_overlap(self, chunks: list[str], max_size: int) -> list[str]:
        """Repeat a trailing sentence when it fits without creating extra chunks."""
        if self.overlap_sentences <= 0 or len(chunks) < 2:
            return chunks

        overlapped = [chunks[0]]
        for chunk in chunks[1:]:
            previous_sentences = self._split_sentences(overlapped[-1])
            overlap = " ".join(previous_sentences[-self.overlap_sentences:]).strip()
            if not overlap or chunk.startswith(overlap):
                overlapped.append(chunk)
                continue

            if len(overlap) + 2 + len(chunk) > max_size:
                overlapped.append(chunk)
                continue
            overlapped.append(f"{overlap}\n\n{chunk}".strip())
        return overlapped

    @staticmethod
    def _split_large_chunk_to_limit(text: str, limit: int) -> list[str]:
        """Sentence-aware splitting with an explicit per-call character limit."""
        if len(text) <= limit:
            return [text]
        parts: list[str] = []
        sentences = re.split(r"(?<=[.!?\n])\s+", text)
        current = ""
        for sentence in sentences:
            if len(sentence) > limit:
                if current:
                    parts.append(current.strip())
                    current = ""
                words = sentence.split()
                segment = ""
                for word in words:
                    if len(word) > limit:
                        if segment:
                            parts.append(segment)
                            segment = ""
                        parts.extend(word[i:i + limit] for i in range(0, len(word), limit))
                    elif len(segment) + len(word) + 1 > limit and segment:
                        parts.append(segment)
                        segment = word
                    else:
                        segment = f"{segment} {word}".strip()
                if segment:
                    parts.append(segment)
                continue
            if len(current) + len(sentence) + 1 > limit and current:
                parts.append(current.strip())
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            parts.append(current.strip())
        return parts or [text]

    def _chunk_section(self, text: str, max_size: int) -> list[str]:
        """Apply semantic boundaries, size normalization, and overlap to a section."""
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return self._split_large_chunk_to_limit(text, max_size)

        similarities = self._compute_similarities(sentences)
        breakpoints = self._find_breakpoints(similarities)
        raw_chunks: list[str] = []
        previous = 0
        for breakpoint in breakpoints:
            value = "\n".join(sentences[previous:breakpoint]).strip()
            if value:
                raw_chunks.append(value)
            previous = breakpoint
        tail = "\n".join(sentences[previous:]).strip()
        if tail:
            raw_chunks.append(tail)

        raw_chunks = self._merge_small_chunks(raw_chunks)
        sized: list[str] = []
        for value in raw_chunks:
            sized.extend(self._split_large_chunk_to_limit(value, max_size))
        return self._add_overlap(sized, max_size)

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
        title, sections = self._split_markdown_sections(text)
        if not sections:
            sections = [([], text)]

        prepared: list[tuple[str, list[str]]] = []
        for section_path, section_text in sections:
            prefix = self._context_prefix(
                title or doc.metadata.get("entry_title", doc.doc_id),
                section_path,
            )
            content_limit = max(100, self.max_chunk_size - len(prefix) - 2)
            section_chunks = self._chunk_section(section_text, content_limit)
            for section_chunk in section_chunks:
                content = f"{prefix}\n\n{section_chunk}".strip() if prefix else section_chunk
                prepared.append((content, section_path))

        chunks: list[Chunk] = []
        for i, (content, section_path) in enumerate(prepared):
            chunks.append(Chunk(
                content=content,
                chunk_id=f"{doc.doc_id}_chunk_{i}",
                doc_id=doc.doc_id,
                document_type=doc.document_type,
                chunk_index=i,
                metadata={
                    **doc.metadata,
                    "section_path": section_path,
                    "section_title": section_path[-1] if section_path else "",
                },
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
