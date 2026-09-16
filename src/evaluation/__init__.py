"""
GaleMed AI - Evaluation Package
Provides RAGAS-based evaluation pipeline for the medical RAG system.
"""

import sys
import types

if "langchain_community.chat_models.vertexai" not in sys.modules:
    _vmod = types.ModuleType("langchain_community.chat_models.vertexai")
    _vmod.ChatVertexAI = object
    sys.modules["langchain_community.chat_models.vertexai"] = _vmod

if "langchain_community.llms.vertexai" not in sys.modules:
    _vmod_llm = types.ModuleType("langchain_community.llms.vertexai")
    _vmod_llm.VertexAI = object
    sys.modules["langchain_community.llms.vertexai"] = _vmod_llm

if "langchain_community.chat_models" in sys.modules:
    setattr(sys.modules["langchain_community.chat_models"], "vertexai", _vmod)

from .dataset import MedicalBenchmarkDataset, load_benchmark

from .ragas_evaluator import RagasEvaluator

__all__ = ["MedicalBenchmarkDataset", "load_benchmark", "RagasEvaluator"]
