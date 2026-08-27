"""
entity_extractor.py
Uses an LLM to extract structured entities and relationships from documents.

Extraction is cached per document to avoid redundant API calls.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI

from src.models import Document
from src.graph.entity_models import (
    ExtractionResult, Policy, Stakeholder, Product,
    Regulation, TechnicalDoc, Relationship,
)


_SYSTEM_PROMPT = """You are an expert knowledge graph builder for enterprise documentation.
Extract all entities and relationships from the provided document.
Return ONLY valid JSON matching the schema below. No explanations, no markdown fences.

Schema:
{
  "policies": [{"policy_id": "POL-XXX", "name": "...", "owner": "...", "effective_date": "...", "review_date": "...", "regulations": ["GDPR", "CCPA"]}],
  "stakeholders": [{"name": "...", "role": "...", "responsibilities": ["..."]}],
  "products": [{"product_id": "...", "name": "...", "category": "...", "version": "...", "features": ["..."]}],
  "regulations": [{"name": "GDPR", "articles": ["Article 5", "Article 17"]}],
  "technical_docs": [{"doc_id": "...", "title": "...", "version": "...", "error_codes": ["ERR_..."], "technologies": ["OAuth 2.0", "JWT"]}],
  "relationships": [
    {"source_id": "POL-001", "source_type": "Policy", "target_id": "Chief Privacy Officer (CPO)", "target_type": "Stakeholder", "relation_type": "OWNED_BY"},
    {"source_id": "POL-001", "source_type": "Policy", "target_id": "GDPR", "target_type": "Regulation", "relation_type": "COMPLIES_WITH"}
  ]
}

Relationship types to use:
- OWNED_BY        : Policy/TechnicalDoc -> Stakeholder (owner)
- COMPLIES_WITH   : Policy -> Regulation
- REFERENCES      : Policy/TechnicalDoc -> Policy/TechnicalDoc (cross-references)
- RESPONSIBLE_FOR : Stakeholder -> Policy/Product (area of responsibility)
- PART_OF         : Product feature -> Product
- RELATES_TO      : any -> any (generic)

Include ONLY entities actually mentioned in the document.
If a field has no data, use an empty string or empty list.
"""


class EntityExtractor:
    """
    Extracts structured entities from documents using an LLM.
    Results are cached per document (keyed by doc_id + content hash).
    """

    def __init__(
        self,
        openai_client: OpenAI,
        model: str,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.client = openai_client
        self.model = model
        self.cache_dir = cache_dir or Path("cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self.cache_dir / "entity_cache.json"
        self._cache: dict[str, dict] = self._load_cache()

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _load_cache(self) -> dict[str, dict]:
        if self._cache_file.exists():
            try:
                return json.loads(self._cache_file.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_cache(self) -> None:
        self._cache_file.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _cache_key(self, doc: Document) -> str:
        content_hash = hashlib.sha256(doc.page_content.encode()).hexdigest()[:12]
        return f"{doc.doc_id}:{content_hash}"

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------

    def extract_from_document(self, doc: Document) -> ExtractionResult:
        """
        Extract entities and relationships from a single document.
        Cached by (doc_id + content hash).
        """
        key = self._cache_key(doc)
        if key in self._cache:
            print(f"  [cache] {doc.doc_id}")
            return self._parse_llm_response(self._cache[key], doc.doc_id)

        print(f"  [LLM]   {doc.doc_id}  ({len(doc.page_content)} chars)  ...", end="", flush=True)
        t = time.time()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Document:\n\n{doc.page_content}"},
            ],
            temperature=0.0,
            max_tokens=1500,
            response_format={"type": "json_object"},
        )

        raw_json = response.choices[0].message.content.strip()
        elapsed = time.time() - t
        print(f"  {elapsed:.1f}s")

        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError as e:
            print(f"  [WARN] JSON parse failed for {doc.doc_id}: {e}")
            parsed = {}

        self._cache[key] = parsed
        self._save_cache()
        return self._parse_llm_response(parsed, doc.doc_id)

    def _parse_llm_response(self, data: dict, doc_id: str) -> ExtractionResult:
        """Convert raw LLM JSON dict into validated Pydantic models."""
        result = ExtractionResult()

        for raw in data.get("policies", []):
            try:
                result.policies.append(Policy(doc_id=doc_id, **raw))
            except Exception:
                pass

        for raw in data.get("stakeholders", []):
            try:
                result.stakeholders.append(Stakeholder(**raw))
            except Exception:
                pass

        for raw in data.get("products", []):
            try:
                result.products.append(Product(doc_id=doc_id, **raw))
            except Exception:
                pass

        for raw in data.get("regulations", []):
            try:
                result.regulations.append(Regulation(**raw))
            except Exception:
                pass

        for raw in data.get("technical_docs", []):
            try:
                result.technical_docs.append(TechnicalDoc(**raw))
            except Exception:
                pass

        for raw in data.get("relationships", []):
            try:
                result.relationships.append(Relationship(**raw))
            except Exception:
                pass

        return result

    # ------------------------------------------------------------------
    # Batch extraction
    # ------------------------------------------------------------------

    def extract_all(self, docs: list[Document]) -> ExtractionResult:
        """
        Extract from all documents, merge and deduplicate results.
        """
        print(f"[EntityExtractor] Extracting from {len(docs)} documents:")
        combined = ExtractionResult()

        for doc in docs:
            doc_result = self.extract_from_document(doc)
            combined.merge(doc_result)

        combined.deduplicate()
        print(f"[EntityExtractor] Done. {combined.summary()}")
        return combined
