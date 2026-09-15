"""
GaleMed AI — Experiments Package
===================================
Ablation study comparing RAG architectures on the 105-question medical benchmark.

Modules:
    architectures  — NaiveRAGPipeline & FullGaleMedRAGPipeline callable wrappers
    runner         — Experiment runner orchestrating RAGAS evaluation per architecture
    analysis       — Statistical comparison + PNG chart generation
"""

from .architectures import NaiveRAGPipeline, FullGaleMedRAGPipeline

__all__ = [
    "NaiveRAGPipeline",
    "FullGaleMedRAGPipeline",
]
