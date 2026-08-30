"""
entity_models.py
Pydantic models for MEDICAL domain entities extracted from
The Gale Encyclopedia of Medicine (3rd Edition).

These models define the schema for nodes and relationships in the Neo4j Knowledge Graph.

Medical Entity Hierarchy:
  - Disease     → Condition/disorder (e.g. Asthma, Diabetes mellitus)
  - Medication  → Drug/substance (e.g. Aspirin, Albuterol)
  - Symptom     → Clinical sign/symptom (e.g. wheezing, chest pain)
  - MedicalProcedure → Diagnostic/therapeutic procedure (e.g. Appendectomy, MRI)
  - MedicalEntry     → A document-level reference linking to the vector store chunk

Relationship types:
  - HAS_SYMPTOM        (Disease → Symptom)
  - TREATED_BY         (Disease → Medication | MedicalProcedure)
  - DIAGNOSED_BY       (Disease → MedicalProcedure)
  - CAUSES_SIDE_EFFECT (Medication → Symptom)
  - CONTRAINDICATED_WITH (Medication → Disease | Medication)
  - INTERACTS_WITH     (Medication → Medication)
  - ALTERNATIVE_TREATMENT (Disease → MedicalProcedure)
  - RELATED_TO         (Disease → Disease | Medication → Disease)
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ── Node Models ──────────────────────────────────────────────────────────────

class Disease(BaseModel):
    """A medical condition, disorder, or disease entity."""
    disease_id: str                                  # e.g. "asthma", "diabetes_mellitus"
    name: str                                        # e.g. "Asthma"
    category: str = ""                               # e.g. "Respiratory", "Endocrine"
    doc_id: str = ""                                 # Reference to vector store doc_id
    icd_codes: list[str] = Field(default_factory=list)   # e.g. ["J45", "E11"]
    symptoms: list[str] = Field(default_factory=list)    # Key symptoms (denormalized)
    causes: list[str] = Field(default_factory=list)      # Main causes/risk factors
    diagnostic_tests: list[str] = Field(default_factory=list)  # Common diagnostic tests


class Medication(BaseModel):
    """A drug, medicine, or therapeutic substance."""
    drug_id: str                                     # e.g. "aspirin", "albuterol"
    name: str                                        # e.g. "Aspirin"
    generic_name: str = ""                           # Generic/INN name
    brand_names: list[str] = Field(default_factory=list)   # e.g. ["Tylenol", "Motrin"]
    drug_class: str = ""                             # e.g. "NSAID", "Beta-agonist"
    doc_id: str = ""
    indications: list[str] = Field(default_factory=list)   # What diseases/conditions it treats
    side_effects: list[str] = Field(default_factory=list)  # Known adverse effects
    contraindications: list[str] = Field(default_factory=list)  # When NOT to use


class Symptom(BaseModel):
    """A clinical sign or symptom experienced by a patient."""
    symptom_id: str                                  # e.g. "wheezing", "chest_pain"
    name: str                                        # e.g. "Wheezing"
    affected_body_part: str = ""                     # e.g. "Respiratory tract", "Chest"
    severity_levels: list[str] = Field(default_factory=list)  # e.g. ["mild", "moderate", "severe"]


class MedicalProcedure(BaseModel):
    """A diagnostic test or therapeutic medical procedure."""
    procedure_id: str                                # e.g. "appendectomy", "mri_scan"
    name: str                                        # e.g. "Appendectomy"
    procedure_type: str = ""                         # e.g. "Surgical", "Diagnostic", "Therapeutic"
    doc_id: str = ""
    purpose: list[str] = Field(default_factory=list)    # What it's used for
    risks: list[str] = Field(default_factory=list)      # Associated risks
    preparation: list[str] = Field(default_factory=list) # Pre-procedure requirements


class MedicalEntry(BaseModel):
    """
    A top-level encyclopedia entry (the Markdown file as a whole).
    Acts as a document node linking the structured graph entities
    back to the raw text chunks in the vector store.
    """
    entry_id: str                                    # e.g. "Asthma", "Aspirin"
    title: str                                       # e.g. "Asthma"
    entry_type: str = ""                             # e.g. "Disease", "Drug", "Procedure", "Test"
    doc_id: str = ""                                 # Vector store doc_id
    related_entries: list[str] = Field(default_factory=list)  # Cross-references


# ── Relationship Model ───────────────────────────────────────────────────────

class Relationship(BaseModel):
    """A directed relationship between two medical entities."""
    source_id: str       # e.g. "asthma", "aspirin"
    source_type: str     # "Disease" | "Medication" | "Symptom" | "MedicalProcedure"
    target_id: str
    target_type: str
    relation_type: str   # "HAS_SYMPTOM" | "TREATED_BY" | "DIAGNOSED_BY" |
                         # "CAUSES_SIDE_EFFECT" | "CONTRAINDICATED_WITH" |
                         # "INTERACTS_WITH" | "ALTERNATIVE_TREATMENT" | "RELATED_TO"
    notes: str = ""      # Optional clarification (e.g. "contraindicated in children")


# ── Extraction Result ────────────────────────────────────────────────────────

class ExtractionResult(BaseModel):
    """All entities and relationships extracted from one or more medical documents."""
    diseases: list[Disease] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    symptoms: list[Symptom] = Field(default_factory=list)
    procedures: list[MedicalProcedure] = Field(default_factory=list)
    entries: list[MedicalEntry] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)

    def merge(self, other: "ExtractionResult") -> None:
        """Merge another extraction result into this one (in-place)."""
        self.diseases.extend(other.diseases)
        self.medications.extend(other.medications)
        self.symptoms.extend(other.symptoms)
        self.procedures.extend(other.procedures)
        self.entries.extend(other.entries)
        self.relationships.extend(other.relationships)

    def deduplicate(self) -> None:
        """Remove duplicate nodes (by ID/name), keeping first occurrence."""
        seen_diseases: set[str] = set()
        seen_medications: set[str] = set()
        seen_symptoms: set[str] = set()
        seen_procedures: set[str] = set()
        seen_entries: set[str] = set()

        self.diseases = [
            d for d in self.diseases
            if d.disease_id not in seen_diseases and not seen_diseases.add(d.disease_id)  # type: ignore
        ]
        self.medications = [
            m for m in self.medications
            if m.drug_id not in seen_medications and not seen_medications.add(m.drug_id)  # type: ignore
        ]
        self.symptoms = [
            s for s in self.symptoms
            if s.symptom_id not in seen_symptoms and not seen_symptoms.add(s.symptom_id)  # type: ignore
        ]
        self.procedures = [
            p for p in self.procedures
            if p.procedure_id not in seen_procedures and not seen_procedures.add(p.procedure_id)  # type: ignore
        ]
        self.entries = [
            e for e in self.entries
            if e.entry_id not in seen_entries and not seen_entries.add(e.entry_id)  # type: ignore
        ]

    def summary(self) -> str:
        return (
            f"Diseases={len(self.diseases)}, "
            f"Medications={len(self.medications)}, "
            f"Symptoms={len(self.symptoms)}, "
            f"Procedures={len(self.procedures)}, "
            f"Entries={len(self.entries)}, "
            f"Relationships={len(self.relationships)}"
        )
