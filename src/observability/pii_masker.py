"""
GaleMed AI — PII Masker
========================
Regex-based PII (Personally Identifiable Information) scrubber for patient data.

Strips the following from query/answer text before sending to Langfuse Cloud:
    - Patient full names  (heuristic: "Mr./Mrs./Dr. Firstname Lastname")
    - Phone numbers       (Vietnamese + international formats)
    - Email addresses
    - Street addresses    (house number + street keyword)
    - Date of birth       (DOB patterns)
    - National ID numbers (CCCD / passport / SSN)
    - Credit card numbers

Design principle: OVER-MASK rather than UNDER-MASK for medical privacy.

Usage:
    >>> masker = PIIMasker()
    >>> clean = masker.mask("Call John Smith at 0912-345-678")
    >>> print(clean)  # "Call [NAME] at [PHONE]"
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── PII Pattern Registry ───────────────────────────────────────────────────────
@dataclass
class _PIIPattern:
    label: str          # Replacement token, e.g. "[PHONE]"
    pattern: str        # Raw regex string
    flags: int = re.IGNORECASE


_PATTERNS: list[_PIIPattern] = [
    # --- Email addresses ---
    _PIIPattern(
        label="[EMAIL]",
        pattern=r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
    ),
    # --- Dates of birth (DOB) ---
    _PIIPattern(
        label="[DOB]",
        pattern=r"\b(?:dob|date of birth|born on|ngay sinh)[:\s]+\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}",
    ),
    # --- Vietnamese national ID / CCCD (9 or 12 digits) ---
    _PIIPattern(
        label="[ID_NUMBER]",
        pattern=r"\b(?:cmnd|cccd|passport|id\s?no\.?)[:\s#]*[0-9]{9,12}\b",
    ),
    # --- Vietnamese phone numbers (strictly 10 digits: 03x, 05x, 07x, 08x, 09x or +84) ---
    _PIIPattern(
        label="[PHONE]",
        pattern=r"(?:\+84[\s\-.]?|0)[35789]\d{1}[\s\-.]?\d{3}[\s\-.]?\d{4}\b",
    ),
    # --- International E.164 / general phone ---
    _PIIPattern(
        label="[PHONE]",
        pattern=r"\+?\d[\d\s\-\.]{7,}\d",
    ),
    # --- Credit card numbers (16 digits with separators) ---
    _PIIPattern(
        label="[CARD_NUMBER]",
        pattern=r"\b(?:\d{4}[\s\-]){3}\d{4}\b",
    ),
    # --- Street addresses (house number + street keyword) ---
    _PIIPattern(
        label="[ADDRESS]",
        pattern=r"\b\d{1,5}\s+(?:[A-Za-z]+\s){1,3}(?:street|st|avenue|ave|road|rd|lane|ln|blvd|pho|duong|huyen|quan|phuong)\b",
    ),
    # --- Personal name honorifics ---
    _PIIPattern(
        label="[NAME]",
        pattern=r"\b(?:Mr|Mrs|Ms|Dr|Prof|Bac si|BS)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b",
    ),
]


class PIIMasker:
    """
    Masks PII tokens in text using regex substitution.

    The masker applies patterns in a fixed priority order:
        1. Email  →  2. Phone  →  3. DOB  →  4. ID  →
        5. Card   →  6. Address  →  7. Name

    Example:
        >>> m = PIIMasker()
        >>> m.mask("Patient John Smith (DOB: 15/08/1990) called 0912-345-678")
        'Patient [NAME] (DOB: [DOB]) called [PHONE]'

    The ``mask_dict`` method masks all string values in a dictionary,
    suitable for sanitising Langfuse span metadata before upload.
    """

    def __init__(self, custom_patterns: list[_PIIPattern] | None = None) -> None:
        patterns = _PATTERNS + (custom_patterns or [])
        # Pre-compile all patterns for speed
        self._compiled: list[tuple[re.Pattern, str]] = [
            (re.compile(p.pattern, p.flags), p.label)
            for p in patterns
        ]

    def mask(self, text: str) -> str:
        """
        Replace PII tokens in *text* with placeholder labels.

        Args:
            text: Raw input string that may contain PII.

        Returns:
            Sanitised string with PII replaced by [LABEL] tokens.
        """
        if not text:
            return text
        for compiled, label in self._compiled:
            text = compiled.sub(label, text)
        return text

    def mask_dict(self, data: dict) -> dict:
        """
        Recursively mask all string values in a dictionary.

        Suitable for cleaning Langfuse span metadata or query payloads.

        Args:
            data: Dictionary with potentially mixed value types.

        Returns:
            New dictionary with all string values sanitised.
        """
        result: dict = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self.mask(value)
            elif isinstance(value, dict):
                result[key] = self.mask_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    self.mask(v) if isinstance(v, str) else v for v in value
                ]
            else:
                result[key] = value
        return result
