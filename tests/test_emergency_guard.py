"""Benchmark regression for the pre-router emergency guard."""

import json
from pathlib import Path

from src.emergency import is_emergency


BENCHMARK = Path(__file__).resolve().parents[1] / "Data/benchmarks/medical_benchmark_with_ground_truth.json"


def test_emergency_guard_matches_benchmark_scope():
    questions = json.loads(BENCHMARK.read_text(encoding="utf-8"))["questions"]
    mismatches = [
        question["id"]
        for question in questions
        if is_emergency(question["query"]) != question["is_emergency"]
    ]
    assert not mismatches, f"Emergency classification disagrees for: {mismatches}"
