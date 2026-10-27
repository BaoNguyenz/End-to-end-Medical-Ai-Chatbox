"""Lightweight emergency guard that runs before retrieval and query routing."""

from __future__ import annotations

import re


# Match descriptions of an acute event, not every educational mention of a
# symptom or treatment. This guard is intentionally independent of the LLM.
_EMERGENCY_PATTERNS = [
    r"\b(?:suicidal|want to die|want to end my life|end my life|kill myself|kill himself|kill herself)\b",
    r"\b(?:severe|crushing|sudden|persistent|ongoing)(?:\s+\w+){0,2}\s+chest pain\b",
    r"\b(?:suspected|having|experiencing)\s+(?:a\s+)?(?:heart attack|stroke)\b",
    r"\b(?:can't|cannot|unable to|inability to|not)\s+breathe\b",
    r"\b(?:severe\s+)?anaphylactic shock\b|\bsevere\s+anaphylaxis\b",
    r"\b(?:massive|suspected)\s+overdose\b",
    r"\b(?:massive|uncontrolled|severe)\s+(?:\w+\s+){0,2}bleeding\b|\bhemorrhage\b",
    r"\b(?:unconscious|unresponsive|choking|drowning|electric shock)\b",
    r"\b(?:acute\s+)?severe\s+asthma attack\b",
    r"\b(?:sudden|severe)\b.{0,60}\babdominal pain\b.{0,60}\b(?:rigid abdomen|high fever)\b",
    r"\b(?:continuous|prolonged)\b.{0,60}\bseizures?\b",
    r"\bseizures?\b.{0,80}\b(?:more than|longer than|over)\s+(?:5|five)\s+minutes\b",
    r"\b(?:severe|extreme)\s+(?:difficulty|trouble)\s+breathing\b",
]


def is_emergency(query: str) -> bool:
    """Detect acute danger signs before any cache lookup or LLM routing."""
    return any(re.search(pattern, query, re.IGNORECASE) for pattern in _EMERGENCY_PATTERNS)
