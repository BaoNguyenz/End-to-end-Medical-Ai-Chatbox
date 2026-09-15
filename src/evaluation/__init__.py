"""
GaleMed AI - Evaluation Package
Provides RAGAS-based evaluation pipeline for the medical RAG system.
"""

from .dataset import MedicalBenchmarkDataset, load_benchmark
from .ragas_evaluator import RagasEvaluator

__all__ = ["MedicalBenchmarkDataset", "load_benchmark", "RagasEvaluator"]
