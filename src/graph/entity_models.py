"""
entity_models.py
Pydantic models for domain entities extracted from TechDocs Inc. documents.

These models define the schema for nodes and relationships in the Neo4j graph.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ── Node Models ────────────────────────────────────────────────────────────

class Policy(BaseModel):
    """A policy document (e.g. POL-001 Data Privacy Policy)."""
    policy_id: str                              # e.g. "POL-001"
    name: str                                   # e.g. "Data Privacy and Protection Policy"
    owner: str                                  # e.g. "Chief Privacy Officer (CPO)"
    effective_date: str = ""                    # e.g. "2024-01-01"
    review_date: str = ""
    doc_id: str = ""                            # reference to vector store doc_id
    regulations: list[str] = Field(default_factory=list)  # e.g. ["GDPR", "CCPA"]


class Stakeholder(BaseModel):
    """A person/role mentioned in a document."""
    name: str                                   # e.g. "Chief Privacy Officer (CPO)"
    role: str = ""                              # e.g. "CPO"
    responsibilities: list[str] = Field(default_factory=list)


class Product(BaseModel):
    """A product in the catalog (e.g. TechDocs Pro)."""
    product_id: str                             # e.g. "TDPRO-2024"
    name: str                                   # e.g. "TechDocs Pro"
    category: str = ""                          # e.g. "Enterprise SaaS"
    version: str = ""
    doc_id: str = ""
    features: list[str] = Field(default_factory=list)


class Regulation(BaseModel):
    """An external regulatory framework (e.g. GDPR)."""
    name: str                                   # e.g. "GDPR"
    articles: list[str] = Field(default_factory=list)  # e.g. ["Article 5", "Article 17"]


class TechnicalDoc(BaseModel):
    """A technical documentation entry."""
    doc_id: str                                 # e.g. "tech_001_api_authentication"
    title: str                                  # e.g. "API Authentication Guide"
    version: str = ""
    error_codes: list[str] = Field(default_factory=list)  # e.g. ["ERR_AUTH_001"]
    technologies: list[str] = Field(default_factory=list) # e.g. ["OAuth 2.0", "JWT"]


# ── Relationship Model ─────────────────────────────────────────────────────

class Relationship(BaseModel):
    """A directed relationship between two nodes."""
    source_id: str          # node property used as ID (policy_id, name, doc_id, etc.)
    source_type: str        # "Policy" | "Stakeholder" | "Product" | "Regulation" | "TechnicalDoc"
    target_id: str
    target_type: str
    relation_type: str      # e.g. "OWNED_BY", "COMPLIES_WITH", "REFERENCES", "RELATES_TO"


# ── Extraction Result ──────────────────────────────────────────────────────

class ExtractionResult(BaseModel):
    """All entities and relationships extracted from one or more documents."""
    policies: list[Policy] = Field(default_factory=list)
    stakeholders: list[Stakeholder] = Field(default_factory=list)
    products: list[Product] = Field(default_factory=list)
    regulations: list[Regulation] = Field(default_factory=list)
    technical_docs: list[TechnicalDoc] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)

    def merge(self, other: "ExtractionResult") -> None:
        """Merge another extraction result into this one (in-place)."""
        self.policies.extend(other.policies)
        self.stakeholders.extend(other.stakeholders)
        self.products.extend(other.products)
        self.regulations.extend(other.regulations)
        self.technical_docs.extend(other.technical_docs)
        self.relationships.extend(other.relationships)

    def deduplicate(self) -> None:
        """Remove duplicate nodes (by ID/name), keeping first occurrence."""
        seen_policies: set[str] = set()
        seen_stakeholders: set[str] = set()
        seen_products: set[str] = set()
        seen_regulations: set[str] = set()
        seen_techdocs: set[str] = set()

        self.policies = [
            p for p in self.policies
            if p.policy_id not in seen_policies and not seen_policies.add(p.policy_id)  # type: ignore
        ]
        self.stakeholders = [
            s for s in self.stakeholders
            if s.name not in seen_stakeholders and not seen_stakeholders.add(s.name)  # type: ignore
        ]
        self.products = [
            p for p in self.products
            if p.product_id not in seen_products and not seen_products.add(p.product_id)  # type: ignore
        ]
        self.regulations = [
            r for r in self.regulations
            if r.name not in seen_regulations and not seen_regulations.add(r.name)  # type: ignore
        ]
        self.technical_docs = [
            t for t in self.technical_docs
            if t.doc_id not in seen_techdocs and not seen_techdocs.add(t.doc_id)  # type: ignore
        ]

    def summary(self) -> str:
        return (
            f"Policies={len(self.policies)}, "
            f"Stakeholders={len(self.stakeholders)}, "
            f"Products={len(self.products)}, "
            f"Regulations={len(self.regulations)}, "
            f"TechnicalDocs={len(self.technical_docs)}, "
            f"Relationships={len(self.relationships)}"
        )
