"""
entity_extractor.py
Uses an LLM to extract structured MEDICAL entities and relationships
from Gale Encyclopedia of Medicine document entries.

Extraction is cached per document to avoid redundant API calls.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI

# pyrefly: ignore [missing-import]
from src.models import Chunk, Document
# pyrefly: ignore [missing-import]
from src.graph.entity_models import (
    ExtractionResult, Disease, Medication, Symptom,
    MedicalProcedure, MedicalEntry, Relationship, EvidenceChunk,
)


_SYSTEM_PROMPT = """\
You are a clinical medical knowledge graph builder.
Analyze the provided medical encyclopedia entry and extract all structured entities and relationships.
Return ONLY valid JSON matching the schema below. No explanations, no markdown fences.

Schema:
{
  "diseases": [
    {
      "disease_id": "lowercase_snake_case_id",
      "name": "Disease Name",
      "category": "e.g. Respiratory / Cardiovascular / Neurological / Infectious",
      "icd_codes": ["J45"],
      "symptoms": ["wheezing", "shortness of breath"],
      "causes": ["allergens", "exercise", "infections"],
      "diagnostic_tests": ["spirometry", "peak flow measurement"]
    }
  ],
  "medications": [
    {
      "drug_id": "lowercase_snake_case_id",
      "name": "Drug Name",
      "generic_name": "generic name",
      "brand_names": ["Tylenol", "Panadol"],
      "drug_class": "NSAID / Beta-agonist / Antibiotic etc.",
      "indications": ["pain relief", "fever reduction"],
      "side_effects": ["nausea", "stomach upset"],
      "contraindications": ["kidney disease", "children under 12"]
    }
  ],
  "symptoms": [
    {
      "symptom_id": "lowercase_snake_case_id",
      "name": "Symptom Name",
      "affected_body_part": "e.g. Lungs / Heart / Brain",
      "severity_levels": ["mild", "moderate", "severe"]
    }
  ],
  "procedures": [
    {
      "procedure_id": "lowercase_snake_case_id",
      "name": "Procedure Name",
      "procedure_type": "Surgical / Diagnostic / Therapeutic",
      "purpose": ["detect cancer", "relieve obstruction"],
      "risks": ["infection", "bleeding"],
      "preparation": ["fasting 8 hours", "discontinue anticoagulants"]
    }
  ],
  "entries": [
    {
      "entry_id": "Title of this encyclopedia entry",
      "title": "Title of this encyclopedia entry",
      "entry_type": "Disease / Drug / Procedure / Test / Therapy",
      "related_entries": ["Asthma", "Bronchitis"]
    }
  ],
  "relationships": [
    {"source_id": "asthma", "source_type": "Disease", "target_id": "wheezing", "target_type": "Symptom", "relation_type": "HAS_SYMPTOM"},
    {"source_id": "asthma", "source_type": "Disease", "target_id": "albuterol", "target_type": "Medication", "relation_type": "TREATED_BY"},
    {"source_id": "aspirin", "source_type": "Medication", "target_id": "stomach_upset", "target_type": "Symptom", "relation_type": "CAUSES_SIDE_EFFECT"},
    {"source_id": "aspirin", "source_type": "Medication", "target_id": "asthma", "target_type": "Disease", "relation_type": "CONTRAINDICATED_WITH", "notes": "can trigger aspirin-sensitive asthma"}
  ]
}

Relationship types to use:
- HAS_SYMPTOM           : Disease → Symptom
- TREATED_BY            : Disease → Medication | MedicalProcedure
- DIAGNOSED_BY          : Disease → MedicalProcedure
- CAUSES_SIDE_EFFECT    : Medication → Symptom
- CONTRAINDICATED_WITH  : Medication → Disease | Medication
- INTERACTS_WITH        : Medication → Medication
- ALTERNATIVE_TREATMENT : Disease → MedicalProcedure (complementary/alternative)
- RELATED_TO            : Disease → Disease | any → any (generic cross-reference)

Rules:
- Include ONLY entities explicitly mentioned in the document.
- Use lowercase_snake_case for all IDs (e.g. "diabetes_mellitus", "aspirin").
- If a field has no data, use an empty string or empty list.
- If the document is not a clinical medical entry (e.g. biography, index page), return all empty lists.
"""


class EntityExtractor:
    """
    Extracts structured medical entities from encyclopedia documents using an LLM.
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
        self._cache_file = self.cache_dir / "medical_entity_cache.json"
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

    def _cache_key(self, item: Document | Chunk) -> str:
        content = item.page_content if isinstance(item, Document) else item.content
        item_id = item.doc_id if isinstance(item, Document) else item.chunk_id
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
        return f"chunk-v1:{item_id}:{content_hash}"

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------

    def extract_from_document(self, doc: Document) -> ExtractionResult:
        """
        Extract medical entities and relationships from a single document.
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
                {"role": "user", "content": f"Medical encyclopedia entry:\n\n{doc.page_content[:4000]}"},
            ],
            temperature=0.0,
            max_tokens=2000,
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

    def extract_from_chunk(self, chunk: Chunk) -> ExtractionResult:
        """Extract entities and relations from one canonical retrieval chunk."""
        key = self._cache_key(chunk)
        if key in self._cache:
            parsed = self._cache[key]
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Document ID: {chunk.doc_id}\n"
                            f"Chunk ID: {chunk.chunk_id}\n\n"
                            f"Medical encyclopedia text unit:\n\n{chunk.content}"
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=2000,
                response_format={"type": "json_object"},
            )
            try:
                parsed = json.loads(response.choices[0].message.content.strip())
            except json.JSONDecodeError:
                parsed = {}
            self._cache[key] = parsed
            self._save_cache()

        result = self._parse_llm_response(parsed, chunk.doc_id)
        result.chunks.append(EvidenceChunk(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            chunk_index=chunk.chunk_index,
        ))
        # Provenance is supplied by trusted application state, never generated
        # by the extraction model.
        for rel in result.relationships:
            rel.source_doc_ids = [chunk.doc_id]
            rel.evidence_chunk_ids = [chunk.chunk_id]
        return result

    def extract_all_chunks(self, chunks: list[Chunk]) -> ExtractionResult:
        """Extract and aggregate graph data from canonical retrieval chunks."""
        print(f"[EntityExtractor] Extracting medical entities from {len(chunks)} chunks:")
        combined = ExtractionResult()
        for chunk in chunks:
            combined.merge(self.extract_from_chunk(chunk))
        combined.deduplicate()
        print(f"[EntityExtractor] Done. {combined.summary()}")
        return combined

    def _parse_llm_response(self, data: dict, doc_id: str) -> ExtractionResult:
        """Convert raw LLM JSON dict into validated Pydantic medical models."""
        result = ExtractionResult()

        for raw in data.get("diseases", []):
            try:
                result.diseases.append(Disease(doc_id=doc_id, **raw))
            except Exception:
                pass

        for raw in data.get("medications", []):
            try:
                result.medications.append(Medication(doc_id=doc_id, **raw))
            except Exception:
                pass

        for raw in data.get("symptoms", []):
            try:
                result.symptoms.append(Symptom(**raw))
            except Exception:
                pass

        for raw in data.get("procedures", []):
            try:
                result.procedures.append(MedicalProcedure(doc_id=doc_id, **raw))
            except Exception:
                pass

        for raw in data.get("entries", []):
            try:
                result.entries.append(MedicalEntry(doc_id=doc_id, **raw))
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
        print(f"[EntityExtractor] Extracting medical entities from {len(docs)} documents:")
        combined = ExtractionResult()

        for doc in docs:
            doc_result = self.extract_from_document(doc)
            combined.merge(doc_result)

        combined.deduplicate()
        print(f"[EntityExtractor] Done. {combined.summary()}")
        return combined
