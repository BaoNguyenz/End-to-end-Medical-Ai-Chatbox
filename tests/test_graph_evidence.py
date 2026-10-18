from src.graph.entity_models import EvidenceChunk, ExtractionResult, Relationship
from src.graph.graph_retriever import GraphHit
from src.models import Chunk, SearchResult, SearchSource
from src.orchestrator.pipeline import RAGPipeline


def _result(chunk_id: str, source: SearchSource) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            content=f"Evidence from {chunk_id}",
            chunk_id=chunk_id,
            doc_id="Aspirin",
            document_type="technical",
        ),
        score=0.5,
        source=source,
    )


def test_relationship_deduplication_preserves_all_evidence_chunks():
    common = {
        "source_id": "aspirin",
        "source_type": "Medication",
        "target_id": "warfarin",
        "target_type": "Medication",
        "relation_type": "INTERACTS_WITH",
    }
    extracted = ExtractionResult(
        chunks=[
            EvidenceChunk(chunk_id="Aspirin_chunk_1", doc_id="Aspirin"),
            EvidenceChunk(chunk_id="Aspirin_chunk_2", doc_id="Aspirin"),
        ],
        relationships=[
            Relationship(
                **common,
                source_doc_ids=["Aspirin"],
                evidence_chunk_ids=["Aspirin_chunk_1"],
            ),
            Relationship(
                **common,
                source_doc_ids=["Aspirin"],
                evidence_chunk_ids=["Aspirin_chunk_2"],
            ),
        ],
    )

    extracted.deduplicate()

    assert len(extracted.relationships) == 1
    assert extracted.relationships[0].evidence_chunk_ids == [
        "Aspirin_chunk_1",
        "Aspirin_chunk_2",
    ]


def test_graph_evidence_is_deduplicated_with_text_candidates():
    text = _result("Aspirin_chunk_1", SearchSource.HYBRID)
    graph = _result("Aspirin_chunk_1", SearchSource.GRAPH)

    merged = RAGPipeline._merge_text_candidates([text], [graph])

    assert len(merged) == 1
    assert merged[0].chunk.metadata["retrieval_sources"] == ["hybrid", "graph"]


def test_only_graph_facts_with_final_evidence_are_formatted():
    hits = [
        GraphHit(
            content="drug: Aspirin | interacts_with: Warfarin",
            evidence_chunk_ids=["Aspirin_chunk_1"],
        ),
        GraphHit(
            content="unsupported fact",
            evidence_chunk_ids=["Aspirin_chunk_9"],
        ),
    ]

    context = RAGPipeline._format_verified_graph_facts(
        hits,
        [_result("Aspirin_chunk_1", SearchSource.HYBRID)],
    )

    assert "interacts_with: Warfarin" in context
    assert "unsupported fact" not in context
