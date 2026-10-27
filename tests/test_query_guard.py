"""Benchmark regression for the pre-router safety guards."""

import json
from pathlib import Path

from src.query_guard import classify_rejection, is_emergency


BENCHMARK = Path(__file__).resolve().parents[1] / "Data/benchmarks/medical_benchmark_with_ground_truth.json"


def test_emergency_guard_matches_benchmark_scope():
    questions = json.loads(BENCHMARK.read_text(encoding="utf-8"))["questions"]
    mismatches = [
        question["id"]
        for question in questions
        if is_emergency(question["query"]) != question["is_emergency"]
    ]
    assert not mismatches, f"Emergency classification disagrees for: {mismatches}"


def test_rejection_guard_matches_benchmark_scope():
    questions = json.loads(BENCHMARK.read_text(encoding="utf-8"))["questions"]
    mismatches = [
        question["id"]
        for question in questions
        if classify_rejection(question["query"])
        != (question["category"] if question["expect_rejection"] else None)
    ]
    assert not mismatches, f"Rejection classification disagrees for: {mismatches}"


def test_medical_questions_with_attack_words_are_not_rejected():
    assert classify_rejection("What are the symptoms of a drug overdose?") is None
    assert classify_rejection("Can aspirin cause bleeding when taken with warfarin?") is None
