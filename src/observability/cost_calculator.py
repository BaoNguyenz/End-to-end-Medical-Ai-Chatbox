"""
GaleMed AI — Cost Calculator
==============================
Converts OpenAI token usage to USD cost using official pricing tables.

Supports:
    - gpt-4o-mini  (primary model)
    - gpt-4o       (fallback / future)
    - text-embedding-3-small (embedding)

Usage:
    >>> calc = CostCalculator()
    >>> usage = TokenUsage(prompt_tokens=500, completion_tokens=200)
    >>> cost = calc.calculate(usage, model="gpt-4o-mini")
    >>> print(f"${cost:.6f}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Pricing table (USD per 1M tokens) — last updated 2025-Q1 ──────────────────
# Source: https://openai.com/api/pricing/
_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {
        "input":  0.150,   # $0.150 per 1M input tokens
        "output": 0.600,   # $0.600 per 1M output tokens
    },
    "gpt-4o": {
        "input":  2.50,    # $2.50 per 1M input tokens
        "output": 10.00,   # $10.00 per 1M output tokens
    },
    "gpt-4o-mini-2024-07-18": {  # alias
        "input":  0.150,
        "output": 0.600,
    },
    "text-embedding-3-small": {
        "input":  0.020,   # $0.020 per 1M tokens
        "output": 0.0,
    },
    "text-embedding-3-large": {
        "input":  0.130,
        "output": 0.0,
    },
}

_DEFAULT_MODEL = "gpt-4o-mini"


@dataclass
class TokenUsage:
    """Token counts from a single LLM API call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @classmethod
    def from_openai_response(cls, response) -> "TokenUsage":
        """
        Build TokenUsage from an openai.types.Completion or ChatCompletion object.

        Example:
            response = client.chat.completions.create(...)
            usage = TokenUsage.from_openai_response(response)
        """
        if hasattr(response, "usage") and response.usage:
            return cls(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
            )
        return cls()


@dataclass
class CostRecord:
    """Accumulated cost over a session or request batch."""

    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    num_calls: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "CostRecord") -> "CostRecord":
        return CostRecord(
            model=self.model,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            num_calls=self.num_calls + other.num_calls,
            errors=self.errors + other.errors,
        )


class CostCalculator:
    """
    Converts OpenAI token usage to USD cost.

    Example:
        >>> calc = CostCalculator()
        >>> usage = TokenUsage(prompt_tokens=1000, completion_tokens=300)
        >>> cost = calc.calculate(usage, model="gpt-4o-mini")
        >>> print(f"${cost:.6f}")  # $0.000330
    """

    def calculate(self, usage: TokenUsage, model: str = _DEFAULT_MODEL) -> float:
        """
        Calculate USD cost for a single API call.

        Args:
            usage: TokenUsage dataclass with prompt/completion counts.
            model: OpenAI model name (must be in pricing table).

        Returns:
            Cost in USD (float). Returns 0.0 if model not found.
        """
        pricing = _PRICING.get(model)
        if pricing is None:
            # Fallback: try to find a matching prefix (e.g. "gpt-4o-mini-...")
            for key in _PRICING:
                if model.startswith(key):
                    pricing = _PRICING[key]
                    break

        if pricing is None:
            logger.warning("No pricing info for model '%s'. Defaulting to $0.", model)
            return 0.0

        input_cost  = (usage.prompt_tokens     / 1_000_000) * pricing["input"]
        output_cost = (usage.completion_tokens  / 1_000_000) * pricing["output"]
        return round(input_cost + output_cost, 8)

    def accumulate(
        self,
        usage: TokenUsage,
        model: str = _DEFAULT_MODEL,
    ) -> CostRecord:
        """
        Build a CostRecord for a single API call.

        Useful for aggregating costs across multiple calls:
            record = calc.accumulate(usage1) + calc.accumulate(usage2)
        """
        cost = self.calculate(usage, model=model)
        return CostRecord(
            model=model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_usd=cost,
            num_calls=1,
        )

    @staticmethod
    def supported_models() -> list[str]:
        """Return list of models with known pricing."""
        return list(_PRICING.keys())

    def format_cost(self, cost_usd: float) -> str:
        """Human-readable cost string."""
        if cost_usd < 0.001:
            return f"${cost_usd * 1000:.4f}m"   # millicents
        return f"${cost_usd:.6f}"


def estimate_cost(
    model: str = "gpt-4o-mini",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> float:
    """Convenience helper to estimate USD cost given token counts."""
    calc = CostCalculator()
    return calc.calculate(
        TokenUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        model=model,
    )


