"""Local Qwen routing and candidate-pool regression tests."""

from types import SimpleNamespace
from pathlib import Path
from uuid import uuid4

from src.transformation.transformation_router import TransformationRouter
from src.transformation.query_decomposition import QueryDecomposer, _DECOMPOSE_PROMPT


class FakeCompletions:
    def __init__(self, content: str):
        self.content = content
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


def make_router(content: str):
    completions = FakeCompletions(content)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    router = TransformationRouter(client, f"local-qwen-test-{uuid4().hex}", object(), Path("cache"))
    return router, completions


def test_qwen_classification_is_cached_per_query():
    router, completions = make_router('{"query_class":"complex"}')

    assert router._classify("Compare two treatments.")[0:2] == ("complex", "qwen")
    assert router._classify("Compare two treatments.")[0:2] == ("complex", "cache")
    assert completions.calls == 1


def test_invalid_qwen_response_falls_back_to_direct_search():
    router, _ = make_router('{"query_class":"graph"}')

    assert router._classify("What are asthma symptoms?")[0:2] == (
        "simple", "fallback_error"
    )


def test_vague_query_keeps_original_hybrid_and_hyde_candidates():
    router, _ = make_router('{"query_class":"vague"}')
    router.hyde.search = lambda *args, **kwargs: (
        [SimpleNamespace(chunk=SimpleNamespace(chunk_id="hyde"))],
        "hypothetical passage",
    )
    hybrid = SimpleNamespace(search=lambda *args, **kwargs: [
        SimpleNamespace(chunk=SimpleNamespace(chunk_id="bm25_vector")),
    ])

    results, meta = router.transform_and_search(
        "chest discomfort after exercise", object(), hybrid, top_k=5
    )

    assert {item.chunk.chunk_id for item in results} == {"bm25_vector", "hyde"}
    assert meta["query_class"] == "vague"
    assert meta["classification_source"] == "qwen"


def test_local_qwen_decomposition_schema_is_parseable():
    prompt = _DECOMPOSE_PROMPT.format(query="Compare two treatments.")
    assert '"sub_questions"' in prompt
    decomposer = object.__new__(QueryDecomposer)
    assert decomposer._parse_response(
        '{"sub_questions": ["How is asthma treated?", "How is COPD treated?"]}',
        "Compare asthma and COPD treatments.",
    ) == ["How is asthma treated?", "How is COPD treated?"]
