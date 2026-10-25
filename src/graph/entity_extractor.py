"""
entity_extractor.py
Uses an LLM to extract structured MEDICAL entities and relationships
from Gale Encyclopedia of Medicine document entries.

Extraction is cached per chunk and model to avoid redundant local inference.
"""

from __future__ import annotations

import hashlib
import json
import re
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
- Treat the supplied text unit as the entire evidence. Do not complete facts from medical knowledge or adjacent sections.
- Every nonempty attribute and every relationship must be explicitly supported by words in this text unit. Omit uncertain facts.
- Every relationship endpoint must also appear in the appropriate entity list above.
- Use ONLY Disease, Medication, Symptom, MedicalProcedure, or MedicalEntry for source_type and target_type.
- Diagnostic tests belong in procedures with type MedicalProcedure; never use Diagnostic_Test as a node type.
- TREATED_BY must point from Disease to Medication or MedicalProcedure. A medication treating a symptom is outside this schema; omit that relation.
- CAUSES_SIDE_EFFECT requires an explicit side effect of the named medication in this text unit. Do not infer from the drug class.
- Use lowercase_snake_case for all IDs (e.g. "diabetes_mellitus", "aspirin").
- If a field has no data, use an empty string or empty list.
- If the document is not a clinical medical entry (e.g. biography, index page), return all empty lists.
"""

_COMPACT_SYSTEM_PROMPT = """/no_think
Extract a SMALL knowledge graph only from this text unit. Return compact JSON using
exactly the six array keys diseases, medications, symptoms, procedures, entries,
and relationships. Each entity needs only its ID and name; omit all optional fields.
Use descriptive lowercase_snake_case IDs, never arbitrary codes such as D1 or M1.
Include only SPECIFIC, explicitly named medical entities. Generic phrases such as
"other medicines", "prescription drugs", and a drug category title are not drugs.
If a sentence announces a list but the list is absent from this text unit, do not
invent its members. A relation requires both specific endpoints and an explicit
link in this text unit. Empty arrays are correct. No explanation or markdown.
"""


def _compact_array(id_key: str, name_key: str = "name", max_items: int = 8) -> dict:
    return {
        "type": "array",
        "maxItems": max_items,
        "items": {
            "type": "object",
            "properties": {id_key: {"type": "string"}, name_key: {"type": "string"}},
            "required": [id_key, name_key],
            "additionalProperties": False,
        },
    }


_COMPACT_SCHEMA = {
    "type": "object",
    "properties": {
        "diseases": _compact_array("disease_id"),
        "medications": _compact_array("drug_id"),
        "symptoms": _compact_array("symptom_id"),
        "procedures": _compact_array("procedure_id"),
        "entries": _compact_array("entry_id", "title", 2),
        "relationships": {
            "type": "array",
            "maxItems": 20,
            "items": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "source_type": {"type": "string", "enum": ["Disease", "Medication", "Symptom", "MedicalProcedure", "MedicalEntry"]},
                    "target_id": {"type": "string"},
                    "target_type": {"type": "string", "enum": ["Disease", "Medication", "Symptom", "MedicalProcedure", "MedicalEntry"]},
                    "relation_type": {"type": "string", "enum": [
                        "HAS_SYMPTOM", "TREATED_BY", "DIAGNOSED_BY", "CAUSES_SIDE_EFFECT",
                        "CONTRAINDICATED_WITH", "INTERACTS_WITH", "ALTERNATIVE_TREATMENT", "RELATED_TO",
                    ]},
                },
                "required": ["source_id", "source_type", "target_id", "target_type", "relation_type"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["diseases", "medications", "symptoms", "procedures", "entries", "relationships"],
    "additionalProperties": False,
}


_RELATION_TYPES = {
    "HAS_SYMPTOM": {("Disease", "Symptom")},
    "TREATED_BY": {("Disease", "Medication"), ("Disease", "MedicalProcedure")},
    "DIAGNOSED_BY": {("Disease", "MedicalProcedure")},
    "CAUSES_SIDE_EFFECT": {("Medication", "Symptom")},
    "CONTRAINDICATED_WITH": {("Medication", "Disease"), ("Medication", "Medication")},
    "INTERACTS_WITH": {("Medication", "Medication")},
    "ALTERNATIVE_TREATMENT": {("Disease", "MedicalProcedure")},
    "RELATED_TO": None,
}

_STOP_WORDS = {"a", "an", "and", "in", "of", "or", "the", "to", "with"}


def _mentioned(value: str, content_words: set[str]) -> bool:
    words = {_word_form(word) for word in re.findall(r"[a-z0-9]+", value.lower())} - _STOP_WORDS
    return bool(words) and words.issubset(content_words)


def _word_form(word: str) -> str:
    if len(word) > 4 and word.endswith("es") and not word.endswith("ses"):
        return word[:-2]
    if len(word) > 4 and word.endswith("s"):
        return word[:-1]
    return word


def _normalize_compact_response(data: dict, doc_id: str, content: str) -> dict:
    """Canonicalize minimal fallback nodes and discard unsupported assertions."""
    words = {_word_form(word) for word in re.findall(r"[a-z0-9]+", content.lower())}
    specs = (
        ("diseases", "Disease", "disease_id", "name"),
        ("medications", "Medication", "drug_id", "name"),
        ("symptoms", "Symptom", "symptom_id", "name"),
        ("procedures", "MedicalProcedure", "procedure_id", "name"),
        ("entries", "MedicalEntry", "entry_id", "title"),
    )
    normalized = {key: [] for key, _, _, _ in specs}
    id_map: dict[tuple[str, str], tuple[str, str]] = {}
    for key, entity_type, id_key, name_key in specs:
        for raw in data.get(key, []):
            name = str(raw.get(name_key, "")).strip()
            old_id = str(raw.get(id_key, "")).strip()
            if not name or not old_id or not _mentioned(name, words):
                continue
            if entity_type == "Medication" and name.lower() in {
                "other medicines", "prescription medicine", "nonprescription medicine",
                "over-the-counter medicine", "antiacne drugs", "drugs",
            }:
                continue
            new_id = doc_id if entity_type == "MedicalEntry" else re.sub(
                r"[^a-z0-9]+", "_", name.lower()
            ).strip("_")
            if not new_id:
                continue
            normalized[key].append({id_key: new_id, name_key: name})
            id_map[(entity_type, old_id)] = (entity_type, new_id)

    type_names = {name.lower(): name for _, name, _, _ in specs}
    normalized["relationships"] = []
    for raw in data.get("relationships", []):
        source_type = type_names.get(str(raw.get("source_type", "")).lower())
        target_type = type_names.get(str(raw.get("target_type", "")).lower())
        relation_type = str(raw.get("relation_type", "")).upper()
        source = id_map.get((source_type, str(raw.get("source_id", ""))))
        target = id_map.get((target_type, str(raw.get("target_id", ""))))
        if not source or not target or relation_type not in _RELATION_TYPES:
            continue
        normalized["relationships"].append({
            "source_id": source[1], "source_type": source[0],
            "target_id": target[1], "target_type": target[0],
            "relation_type": relation_type,
        })
    normalized["__compact"] = True
    return normalized


def _sanitize_extraction_payload(data: dict) -> tuple[dict, int]:
    """Drop malformed list items while preserving usable cached entities."""
    fields = ("diseases", "medications", "symptoms", "procedures", "entries", "relationships")
    cleaned = dict(data)
    dropped = 0
    for field in fields:
        items = data.get(field, [])
        if not isinstance(items, list):
            cleaned[field] = []
            dropped += 1
            continue
        valid = [item for item in items if isinstance(item, dict)]
        cleaned[field] = valid
        dropped += len(items) - len(valid)
    return cleaned, dropped


class EntityExtractor:
    """
    Extracts structured medical entities from encyclopedia documents using an LLM.
    Results are cached by model, chunk ID, and content hash.
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
        model_slug = re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")
        self._cache_file = self.cache_dir / f"medical_entity_cache_{model_slug}_v2.json"
        self._cache_entries_dir = self.cache_dir / f"medical_entity_cache_{model_slug}_v2_chunks"
        self._cache_entries_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, dict] = self._load_cache()

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _load_cache(self) -> dict[str, dict]:
        cache: dict[str, dict] = {}
        # The old whole-file cache remains readable. A failed Windows replace
        # may leave a valid .tmp with one more completed extraction.
        for path in (self._cache_file, self._cache_file.with_suffix(".json.tmp")):
            if not path.exists():
                continue
            try:
                entries = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(entries, dict):
                    for key, value in entries.items():
                        cache.setdefault(key, value)
            except (OSError, json.JSONDecodeError) as error:
                print(f"  [cache] Cannot read {path.name}: {error}", flush=True)

        for path in sorted(self._cache_entries_dir.iterdir()):
            if not path.name.endswith((".json", ".json.tmp")):
                continue
            try:
                entry = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(entry, dict) and isinstance(entry.get("key"), str):
                    cache[entry["key"]] = entry["value"]
            except (OSError, json.JSONDecodeError, KeyError) as error:
                print(f"  [cache] Cannot read {path.name}: {error}", flush=True)
        return cache

    def _save_cache(self, key: str) -> None:
        # One small file per chunk avoids repeatedly replacing the shared
        # multi-megabyte cache, which Windows may temporarily lock for reads.
        destination = self._cache_entries_dir / f"{hashlib.sha256(key.encode()).hexdigest()}.json"
        temporary_file = destination.with_suffix(".json.tmp")
        temporary_file.write_text(
            json.dumps({"key": key, "value": self._cache[key]}, ensure_ascii=False),
            encoding="utf-8",
        )
        for attempt in range(5):
            try:
                temporary_file.replace(destination)
                return
            except PermissionError:
                time.sleep(0.2 * (attempt + 1))
        print(f"  [cache] Keeping recoverable temporary entry: {temporary_file.name}", flush=True)

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
                {"role": "system", "content": _SYSTEM_PROMPT + "\n/no_think"},
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
        self._save_cache(key)
        return self._parse_llm_response(parsed, doc.doc_id)

    def extract_from_chunk(self, chunk: Chunk) -> ExtractionResult:
        """Extract entities and relations from one canonical retrieval chunk."""
        key = self._cache_key(chunk)
        if key in self._cache:
            parsed = self._cache[key]
            if not isinstance(parsed, dict):
                raise ValueError(f"Cached extraction is not a JSON object for {chunk.chunk_id}")
            parsed, dropped = _sanitize_extraction_payload(parsed)
            if dropped:
                print(f"    [cache] ignored {dropped} malformed item(s) in {chunk.chunk_id}", flush=True)
            if parsed.get("__compact") or any(
                str(item.get("drug_id", "")).startswith("drug_")
                for item in parsed.get("medications", [])
            ):
                parsed = _normalize_compact_response(parsed, chunk.doc_id, chunk.content)
        else:
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT + "\n/no_think"},
                {
                    "role": "user",
                    "content": (
                        f"Document ID: {chunk.doc_id}\n"
                        f"Chunk ID: {chunk.chunk_id}\n\n"
                        f"Medical encyclopedia text unit:\n\n{chunk.content}"
                    ),
                },
            ]
            last_error = ""
            attempts = (
                (messages, 4096, {"type": "json_object"}),
                (
                    [{"role": "system", "content": _COMPACT_SYSTEM_PROMPT}, messages[1]],
                    2048,
                    {"type": "json_object", "schema": _COMPACT_SCHEMA},
                ),
                (
                    [{"role": "system", "content": _COMPACT_SYSTEM_PROMPT}, messages[1]],
                    4096,
                    {"type": "json_object", "schema": _COMPACT_SCHEMA},
                ),
            )
            if chunk.content.rstrip().endswith(":"):
                attempts = attempts[1:]
            for attempt_messages, max_tokens, response_format in attempts:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=attempt_messages,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )
                choice = response.choices[0]
                raw = choice.message.content or ""
                if choice.finish_reason == "length":
                    last_error = f"output reached {max_tokens} tokens"
                else:
                    try:
                        parsed = json.loads(raw)
                        if isinstance(parsed, dict):
                            parsed, dropped = _sanitize_extraction_payload(parsed)
                            if dropped:
                                print(f"    [LLM] ignored {dropped} malformed item(s) in {chunk.chunk_id}", flush=True)
                            if response_format.get("schema") is not None:
                                parsed = _normalize_compact_response(parsed, chunk.doc_id, chunk.content)
                            break
                        last_error = "response was not a JSON object"
                    except json.JSONDecodeError as error:
                        last_error = f"invalid JSON: {error}"
                print(f"    [retry] {chunk.chunk_id}: {last_error}", flush=True)
            else:
                raise ValueError(
                    f"Local extraction failed for {chunk.chunk_id} after 3 attempts: {last_error}. "
                    "Previous chunks remain in the extraction cache."
                )
            self._cache[key] = parsed
            self._save_cache(key)

        result = self._parse_llm_response(parsed, chunk.doc_id, chunk.content)
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
        for index, chunk in enumerate(chunks, start=1):
            print(f"  [{index}/{len(chunks)}] {chunk.chunk_id}", flush=True)
            combined.merge(self.extract_from_chunk(chunk))
        combined.deduplicate()
        print(f"[EntityExtractor] Done. {combined.summary()}")
        return combined

    def _parse_llm_response(self, data: dict, doc_id: str, content: str | None = None) -> ExtractionResult:
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

        if content is not None:
            content_words = {_word_form(word) for word in re.findall(r"[a-z0-9]+", content.lower())}
            result.medications = [
                e for e in result.medications
                if _mentioned(e.name, content_words) or _mentioned(e.drug_id, content_words)
            ]
            result.symptoms = [
                e for e in result.symptoms
                if _mentioned(e.name, content_words) or _mentioned(e.symptom_id, content_words)
            ]
            result.procedures = [
                e for e in result.procedures
                if _mentioned(e.name, content_words) or _mentioned(e.procedure_id, content_words)
            ]
            list_fields = (
                [(e, ("icd_codes", "symptoms", "causes", "diagnostic_tests")) for e in result.diseases]
                + [(e, ("brand_names", "indications", "side_effects", "contraindications")) for e in result.medications]
                + [(e, ("severity_levels",)) for e in result.symptoms]
                + [(e, ("purpose", "risks", "preparation")) for e in result.procedures]
                + [(e, ("related_entries",)) for e in result.entries]
            )
            for entity, fields in list_fields:
                for field in fields:
                    setattr(entity, field, [
                        value for value in getattr(entity, field)
                        if _mentioned(value, content_words)
                    ])
            scalar_fields = (
                [(e, ("category",)) for e in result.diseases]
                + [(e, ("generic_name", "drug_class")) for e in result.medications]
                + [(e, ("affected_body_part",)) for e in result.symptoms]
                + [(e, ("procedure_type",)) for e in result.procedures]
            )
            for entity, fields in scalar_fields:
                for field in fields:
                    if getattr(entity, field) and not _mentioned(getattr(entity, field), content_words):
                        setattr(entity, field, "")
            entity_ids = {
                "Disease": {e.disease_id for e in result.diseases},
                "Medication": {e.drug_id for e in result.medications},
                "Symptom": {e.symptom_id for e in result.symptoms},
                "MedicalProcedure": {e.procedure_id for e in result.procedures},
                "MedicalEntry": {e.entry_id for e in result.entries},
            }
            grounded = []
            for rel in result.relationships:
                if rel.source_type == "Procedure":
                    rel.source_type = "MedicalProcedure"
                if rel.target_type == "Procedure":
                    rel.target_type = "MedicalProcedure"
                allowed_pairs = _RELATION_TYPES.get(rel.relation_type, set())
                if allowed_pairs is not None and (rel.source_type, rel.target_type) not in allowed_pairs:
                    continue
                if not _mentioned(rel.source_id, content_words):
                    continue
                if not _mentioned(rel.target_id, content_words):
                    continue
                if rel.source_type not in entity_ids or rel.target_type not in entity_ids:
                    continue
                for entity_type, entity_id in (
                    (rel.source_type, rel.source_id),
                    (rel.target_type, rel.target_id),
                ):
                    if entity_id in entity_ids[entity_type]:
                        continue
                    display_name = entity_id.replace("_", " ")
                    if entity_type == "Disease":
                        result.diseases.append(Disease(disease_id=entity_id, name=display_name, doc_id=doc_id))
                    elif entity_type == "Medication":
                        result.medications.append(Medication(drug_id=entity_id, name=display_name, doc_id=doc_id))
                    elif entity_type == "Symptom":
                        result.symptoms.append(Symptom(symptom_id=entity_id, name=display_name))
                    elif entity_type == "MedicalProcedure":
                        result.procedures.append(MedicalProcedure(procedure_id=entity_id, name=display_name, doc_id=doc_id))
                    elif entity_type == "MedicalEntry":
                        result.entries.append(MedicalEntry(entry_id=entity_id, title=display_name, doc_id=doc_id))
                    entity_ids[entity_type].add(entity_id)
                grounded.append(rel)
            result.relationships = grounded

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
