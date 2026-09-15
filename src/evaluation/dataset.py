"""
GaleMed AI - Medical Benchmark Dataset Loader
==============================================
Loads the 105-question medical benchmark from JSON and converts it to
a HuggingFace Dataset format ready for RAGAS evaluation.

Dataset Structure (per question):
    - id: Unique question identifier (e.g., "R01", "C01")
    - category: Medical specialty (respiratory, cardiovascular, etc.)
    - query_type: simple | complex | vague | adversarial
    - question_type: Factual | Relational | Multi-hop | Analytical
    - expect_rejection: bool — True for jailbreak/harmful queries
    - is_emergency: bool — True for emergency medical situations
    - query: The question text
    - ground_truth: Reference answer from The Gale Encyclopedia of Medicine

Template Ground Truth Detection:
    Some questions still have placeholder ground truths in the format:
    "According to The Gale Encyclopedia of Medicine, ... involves specific
    physiological etiologies, characteristic clinical signs..."
    These are flagged and excluded from context_recall scoring.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Template detection ───────────────────────────────────────────────────────
_TEMPLATE_INDICATORS = [
    "involves specific physiological etiologies, characteristic clinical signs, diagnostic evaluation, and evidence-based therapeutic management protocols",
    "According to The Gale Encyclopedia of Medicine,",
]


def _is_template_ground_truth(text: str) -> bool:
    """Return True if the ground truth is a generic placeholder template."""
    return any(indicator in text for indicator in _TEMPLATE_INDICATORS)


# ── Data classes ─────────────────────────────────────────────────────────────
@dataclass
class BenchmarkQuestion:
    """A single benchmark question with metadata."""

    id: str
    category: str
    query_type: str  # simple | complex | vague | adversarial
    question_type: str  # Factual | Relational | Multi-hop | Analytical
    expect_rejection: bool
    is_emergency: bool
    query: str
    ground_truth: str
    has_template_ground_truth: bool = field(default=False, init=False)

    def __post_init__(self):
        self.has_template_ground_truth = _is_template_ground_truth(self.ground_truth)

    def is_evaluable(self) -> bool:
        """
        Return True if this question is fully evaluable by RAGAS.

        Questions that bypass normal retrieval (emergency alerts, rejected
        jailbreaks) are still evaluable for Safety metrics; they just skip
        context_precision / context_recall scoring.
        """
        return not self.has_template_ground_truth

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "category": self.category,
            "query_type": self.query_type,
            "question_type": self.question_type,
            "expect_rejection": self.expect_rejection,
            "is_emergency": self.is_emergency,
            "query": self.query,
            "ground_truth": self.ground_truth,
            "has_template_ground_truth": self.has_template_ground_truth,
        }


@dataclass
class MedicalBenchmarkDataset:
    """
    Container for the full 105-question medical benchmark.

    Attributes:
        version: Dataset schema version
        questions: All benchmark questions
    """

    version: str
    questions: list[BenchmarkQuestion]

    # ── Properties ────────────────────────────────────────────────────────────
    @property
    def total(self) -> int:
        return len(self.questions)

    @property
    def evaluable_questions(self) -> list[BenchmarkQuestion]:
        """Questions with proper ground truth (no template placeholders)."""
        return [q for q in self.questions if q.is_evaluable()]

    @property
    def template_questions(self) -> list[BenchmarkQuestion]:
        """Questions that still have placeholder ground truths."""
        return [q for q in self.questions if q.has_template_ground_truth]

    @property
    def emergency_questions(self) -> list[BenchmarkQuestion]:
        return [q for q in self.questions if q.is_emergency]

    @property
    def rejection_questions(self) -> list[BenchmarkQuestion]:
        return [q for q in self.questions if q.expect_rejection]

    # ── Filtering helpers ──────────────────────────────────────────────────────
    def filter_by_type(self, question_type: str) -> list[BenchmarkQuestion]:
        """Filter by question_type: Factual | Relational | Multi-hop | Analytical."""
        return [q for q in self.questions if q.question_type == question_type]

    def filter_by_category(self, category: str) -> list[BenchmarkQuestion]:
        """Filter by medical specialty (e.g., 'respiratory', 'cardiovascular')."""
        return [q for q in self.questions if q.category == category]

    def get_sample(self, n: int = 5, *, evaluable_only: bool = True) -> list[BenchmarkQuestion]:
        """Return a small sample for quick smoke testing."""
        pool = self.evaluable_questions if evaluable_only else self.questions
        return pool[:n]

    def print_summary(self) -> None:
        """Print a human-readable dataset summary to stdout."""
        template_count = len(self.template_questions)
        evaluable_count = len(self.evaluable_questions)

        print("\n" + "=" * 60)
        print(f"  GaleMed Medical Benchmark — v{self.version}")
        print("=" * 60)
        print(f"  Total questions      : {self.total}")
        print(f"  Evaluable (w/ GT)    : {evaluable_count}")
        print(f"  Template placeholders: {template_count}")
        print(f"  Emergency alerts     : {len(self.emergency_questions)}")
        print(f"  Rejection / Jailbreak: {len(self.rejection_questions)}")

        # Question type distribution
        print("\n  Distribution by question type:")
        for qtype in ["Factual", "Relational", "Multi-hop", "Analytical"]:
            count = len(self.filter_by_type(qtype))
            pct = count / self.total * 100 if self.total else 0
            bar = "█" * int(pct / 5)
            print(f"    {qtype:<12} {count:>3} ({pct:5.1f}%) {bar}")

        # Category distribution
        print("\n  Distribution by medical specialty:")
        categories: dict[str, int] = {}
        for q in self.questions:
            categories[q.category] = categories.get(q.category, 0) + 1
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            print(f"    {cat:<20} {count:>3} questions")

        print("=" * 60 + "\n")

    # ── HuggingFace Dataset conversion ────────────────────────────────────────
    def to_hf_dataset(
        self,
        *,
        evaluable_only: bool = True,
        include_columns: Optional[list[str]] = None,
    ):
        """
        Convert to a HuggingFace ``datasets.Dataset``.

        Args:
            evaluable_only: If True, only include questions with proper ground truths.
            include_columns: Subset of columns to include. Defaults to all.

        Returns:
            A ``datasets.Dataset`` object with columns:
            ``question``, ``ground_truth``, ``id``, ``category``,
            ``question_type``, ``expect_rejection``, ``is_emergency``.
        """
        try:
            from datasets import Dataset  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required. Install it with: uv add datasets"
            ) from exc

        pool = self.evaluable_questions if evaluable_only else self.questions

        rows = {
            "question": [q.query for q in pool],
            "ground_truth": [q.ground_truth for q in pool],
            "id": [q.id for q in pool],
            "category": [q.category for q in pool],
            "question_type": [q.question_type for q in pool],
            "expect_rejection": [q.expect_rejection for q in pool],
            "is_emergency": [q.is_emergency for q in pool],
        }

        if include_columns:
            rows = {k: v for k, v in rows.items() if k in include_columns}

        return Dataset.from_dict(rows)


# ── Factory functions ─────────────────────────────────────────────────────────
def load_benchmark(
    path: Optional[str | Path] = None,
    *,
    verbose: bool = True,
) -> MedicalBenchmarkDataset:
    """
    Load the medical benchmark JSON and return a ``MedicalBenchmarkDataset``.

    Args:
        path: Path to the benchmark JSON file. Defaults to the canonical
              location in the project: ``Data/benchmarks/medical_benchmark_with_ground_truth.json``.
        verbose: If True, print a summary table after loading.

    Returns:
        A fully populated ``MedicalBenchmarkDataset`` instance.

    Example:
        >>> from src.evaluation.dataset import load_benchmark
        >>> dataset = load_benchmark()
        >>> print(f"Loaded {dataset.total} questions")
    """
    if path is None:
        # Resolve relative to project root (2 parents up from src/evaluation/)
        project_root = Path(__file__).resolve().parent.parent.parent
        path = project_root / "Data" / "benchmarks" / "medical_benchmark_with_ground_truth.json"

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")

    logger.info("Loading benchmark from: %s", path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    questions: list[BenchmarkQuestion] = []
    for item in raw.get("questions", []):
        try:
            q = BenchmarkQuestion(
                id=item["id"],
                category=item.get("category", "unknown"),
                query_type=item.get("query_type", "simple"),
                question_type=item.get("question_type", "Factual"),
                expect_rejection=bool(item.get("expect_rejection", False)),
                is_emergency=bool(item.get("is_emergency", False)),
                query=item["query"],
                ground_truth=item.get("ground_truth", ""),
            )
            questions.append(q)
        except (KeyError, TypeError) as exc:
            logger.warning("Skipping malformed question entry %s: %s", item.get("id", "?"), exc)

    dataset = MedicalBenchmarkDataset(
        version=raw.get("version", "unknown"),
        questions=questions,
    )

    if verbose:
        dataset.print_summary()

    template_count = len(dataset.template_questions)
    if template_count > 0:
        logger.warning(
            "%d questions have placeholder ground truths and will be excluded "
            "from context_recall scoring. Consider updating the benchmark JSON.",
            template_count,
        )

    return dataset


def load_benchmark_dataset(
    path: Optional[str | Path] = None,
    limit: Optional[int] = None,
    evaluable_only: bool = False,
) -> list[dict]:
    """
    Load benchmark dataset as a list of dictionaries with standard keys:
    'id', 'question', 'category', 'query_type', 'question_type', 'ground_truth'.
    """
    dataset = load_benchmark(path, verbose=False)
    pool = dataset.evaluable_questions if evaluable_only else dataset.questions
    if limit is not None and limit > 0:
        pool = pool[:limit]

    return [
        {
            "id": q.id,
            "question": q.query,
            "category": q.category,
            "query_type": q.query_type,
            "question_type": q.question_type,
            "expect_rejection": q.expect_rejection,
            "is_emergency": q.is_emergency,
            "ground_truth": q.ground_truth,
            "has_template_ground_truth": q.has_template_ground_truth,
        }
        for q in pool
    ]


def get_dataset_statistics(data: list[dict] | MedicalBenchmarkDataset) -> dict:
    """Return summary statistics of the dataset."""
    if isinstance(data, MedicalBenchmarkDataset):
        items = [q.to_dict() for q in data.questions]
    else:
        items = data

    total = len(items)
    categories: dict[str, int] = {}
    qtypes: dict[str, int] = {}
    emergencies = 0
    rejections = 0

    for it in items:
        cat = it.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
        qt = it.get("question_type", "unknown")
        qtypes[qt] = qtypes.get(qt, 0) + 1
        if it.get("is_emergency"):
            emergencies += 1
        if it.get("expect_rejection"):
            rejections += 1

    return {
        "total_questions": total,
        "categories": categories,
        "question_types": qtypes,
        "emergency_count": emergencies,
        "rejection_count": rejections,
    }

