from src.indexing.semantic_chunker import SemanticChunker
from src.models import Document


class DeterministicChunker(SemanticChunker):
    def __init__(self, **kwargs):
        super().__init__(_model=object(), **kwargs)

    def _compute_similarities(self, sentences):
        # Force size-based splitting so tests do not load an embedding model.
        return [1.0] * max(0, len(sentences) - 1)


def _document(text: str) -> Document:
    return Document(
        page_content=text,
        metadata={
            "doc_id": "Aspirin",
            "document_type": "technical",
            "entry_title": "Aspirin",
        },
    )


def test_chunks_do_not_cross_markdown_sections_and_keep_heading_context():
    chunker = DeterministicChunker(
        similarity_threshold=0.60,
        min_chunk_size=40,
        max_chunk_size=180,
        overlap_sentences=1,
    )
    doc = _document(
        "# Aspirin\n\n"
        "### Uses\n\nAspirin relieves minor pain. It also reduces fever. "
        "A clinician may recommend it for selected cardiovascular uses.\n\n"
        "### Interactions\n\nAspirin can interact with warfarin. "
        "The combination can increase excessive bleeding risk."
    )

    chunks = chunker.chunk_document(doc)

    assert chunks
    assert all(len(chunk.content) <= 180 for chunk in chunks)
    assert any(chunk.metadata["section_title"] == "Uses" for chunk in chunks)
    assert any(chunk.metadata["section_title"] == "Interactions" for chunk in chunks)
    assert all(
        not ("— Uses" in chunk.content and "— Interactions" in chunk.content)
        for chunk in chunks
    )


def test_overlap_repeats_one_complete_trailing_sentence():
    chunker = DeterministicChunker(
        similarity_threshold=0.60,
        min_chunk_size=20,
        max_chunk_size=100,
        overlap_sentences=1,
    )

    chunks = chunker._add_overlap(
        ["First sentence. Shared trailing sentence.", "New information follows."],
        max_size=100,
    )

    assert chunks[1].startswith("Shared trailing sentence.")
