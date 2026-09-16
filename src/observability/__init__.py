"""
GaleMed AI — Observability Package
====================================
Provides LLM observability, cost tracking, and PII masking.

Modules:
    langfuse_tracker  — Langfuse Cloud trace spans + MockTracer fallback
    cost_calculator   — Token → USD cost calculator (gpt-4o-mini pricing)
    pii_masker        — Regex-based PII scrubber for patient data
"""

from .langfuse_tracker import get_tracer, LangfuseTracer, MockTracer
from .cost_calculator import CostCalculator, TokenUsage
from .pii_masker import PIIMasker

__all__ = [
    "get_tracer",
    "LangfuseTracer",
    "MockTracer",
    "CostCalculator",
    "TokenUsage",
    "PIIMasker",
]
