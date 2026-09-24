"""
tests/test_api.py
Automated unit & integration test suite for GaleMed AI.

Covers:
  - PIIMasker (regex scrubbing of patient data)
  - Emergency Guardrail detection
  - QueryRouter (intent-based search classification)
  - CostCalculator (token counting and pricing)
  - FastAPI endpoints (/api/health, /api/cache/stats)
"""

import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import re
import pytest
from fastapi.testclient import TestClient

from app import app
from src.observability.pii_masker import PIIMasker
from src.retrieval.query_router import QueryRouter
from src.observability.cost_calculator import CostCalculator, TokenUsage
from src.models import QueryType, SearchStrategy
from src.orchestrator.pipeline import _EMERGENCY_PATTERNS



# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def masker():
    return PIIMasker()


@pytest.fixture
def router():
    return QueryRouter()


@pytest.fixture
def cost_calc():
    return CostCalculator()


# ── 1. Unit Tests: PIIMasker ──────────────────────────────────────────────────

def test_pii_masker_email(masker):
    text = "Please send clinical records to patient.john@example.com immediately."
    masked = masker.mask(text)
    assert "[EMAIL]" in masked
    assert "patient.john@example.com" not in masked


def test_pii_masker_phone_vn(masker):
    text = "Patient emergency phone is 0912345678 or +84987654321."
    masked = masker.mask(text)
    assert "[PHONE]" in masked
    assert "0912345678" not in masked
    assert "+84987654321" not in masked


def test_pii_masker_id_and_dob(masker):
    text = "Patient CCCD: 012345678901 and DOB: 15/08/1990."
    masked = masker.mask(text)
    assert "[ID_NUMBER]" in masked
    assert "[DOB]" in masked


def test_pii_masker_dict(masker):
    payload = {
        "query": "Patient Mr. David Miller has high fever",
        "email": "david@med.org",
        "nested": {"phone": "0988123456"}
    }
    masked_dict = masker.mask_dict(payload)
    assert "[NAME]" in masked_dict["query"]
    assert "[EMAIL]" in masked_dict["email"]
    assert "[PHONE]" in masked_dict["nested"]["phone"]


# ── 2. Unit Tests: Emergency Guardrail ─────────────────────────────────────────

def is_emergency_check(query: str) -> bool:
    query_lower = query.lower()
    for pattern in _EMERGENCY_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            return True
    return False


def test_emergency_guardrail_triggers():
    assert is_emergency_check("Patient has severe crushing chest pain and difficulty breathing") is True
    assert is_emergency_check("Suspected stroke and unconscious in emergency room") is True
    assert is_emergency_check("Severe bleeding from artery, anaphylaxis shock") is True


def test_emergency_guardrail_non_emergencies():
    assert is_emergency_check("What are common symptoms of seasonal asthma?") is False
    assert is_emergency_check("Compare therapeutic uses of Aspirin vs Acetaminophen") is False
    assert is_emergency_check("What is the standard dosage range of Ibuprofen?") is False


# ── 3. Unit Tests: QueryRouter Intent Classification ──────────────────────────

def test_query_router_keyword(router):
    # Dosage and lab codes trigger BM25
    q_type, strategy = router.classify("What is the recommended dose of 500mg Amoxicillin?")
    assert q_type == QueryType.KEYWORD
    assert strategy == SearchStrategy.BM25


def test_query_router_semantic(router):
    # Symptom and mechanism queries trigger Vector Search
    q_type, strategy = router.classify("What causes shortness of breath and wheezing in children?")
    assert q_type == QueryType.SEMANTIC
    assert strategy == SearchStrategy.VECTOR


def test_query_router_graph(router):
    # Drug interactions trigger Graph/Complex
    q_type, strategy = router.classify("What are the drug interactions and contraindications of Warfarin?")
    assert q_type == QueryType.COMPLEX


# ── 4. Unit Tests: CostCalculator ─────────────────────────────────────────────

def test_cost_calculator_pricing(cost_calc):
    # GPT-4o-mini pricing calculation
    usage = TokenUsage(prompt_tokens=1000, completion_tokens=1000)
    cost_usd = cost_calc.calculate(usage, model="gpt-4o-mini")
    assert usage.prompt_tokens == 1000
    assert usage.completion_tokens == 1000
    assert usage.total_tokens == 2000
    assert cost_usd > 0.0



# ── 5. Integration Tests: FastAPI Endpoints ────────────────────────────────────

def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "components" in data


def test_cache_stats_endpoint(client):
    response = client.get("/api/cache/stats")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
