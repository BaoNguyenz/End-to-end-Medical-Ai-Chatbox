"""Lightweight emergency and rejection guards run before retrieval/routing."""

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


# This deterministic gate handles explicit prompt attacks and obvious requests
# outside the medical encyclopedia's scope. It does not infer intent from every
# possible phrasing; future semantic routing should keep this guard in front.
_ADVERSARIAL_PATTERNS = [
    r"\b(?:ignore|forget|override)\b.{0,65}\b(?:instructions?|rules?|system|safety|restrictions?)\b",
    r"\b(?:dan|do anything now|unrestricted doctor|no ethical filters|zero safety restrictions)\b",
    r"\b(?:system prompt|system override|new system instruction|\[inst\]|\[/inst\])\b",
    r"\b(?:manufacture illicit|lethal dose|chlorine gas|painless way to overdose|poison extraction)\b",
    r"\b(?:forge an official medical prescription|black-market pharmaceutical|smuggle them)\b",
    r"\[source:[^\]]+\].{0,160}\b(?:confirm|advise the patient)\b",
]

_OUT_OF_SCOPE_PATTERNS = [
    r"\b(?:stock market price|stock price|share price|ticker symbol)\b",
    r"\b(?:python function|quicksort|sort an array|write (?:me )?code)\b",
    r"\b(?:tourist landmarks?|hotels? to visit|fifa world cup|world cup tournament)\b",
    r"\b(?:engine oil|toyota corolla|romantic poem|creative poem|recipe for baking|brownies)\b",
    r"\b(?:quantum superposition|schrodinger'?s cat)\b",
    r"\b(?:fictional virus|fictional miracle drug|sci-fi movie)\b",
]


_REJECTION_RESPONSE = (
    "I can only provide information from The Gale Encyclopedia of Medicine. "
    "Please ask a medical question."
)


def classify_rejection(query: str) -> str | None:
    """Return adversarial/out_of_scope for explicit cases, otherwise None."""
    if any(re.search(pattern, query, re.IGNORECASE) for pattern in _ADVERSARIAL_PATTERNS):
        return "adversarial"
    if any(re.search(pattern, query, re.IGNORECASE) for pattern in _OUT_OF_SCOPE_PATTERNS):
        return "out_of_scope"
    return None
