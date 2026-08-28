"""
knowledge_graph.py
Neo4j Knowledge Graph manager for Medical AI Chatbot.

Manages connections to Neo4j, populates medical entity nodes,
creates relationships, and provides a Cypher query interface.

Medical Entity Labels:
  - Disease
  - Medication
  - Symptom
  - MedicalProcedure
  - MedicalEntry  (document-level anchor to vector store)

Relationship Types:
  - HAS_SYMPTOM, TREATED_BY, DIAGNOSED_BY
  - CAUSES_SIDE_EFFECT, CONTRAINDICATED_WITH, INTERACTS_WITH
  - ALTERNATIVE_TREATMENT, RELATED_TO
"""

from __future__ import annotations

from typing import Optional

from neo4j import GraphDatabase, Driver

from src.graph.entity_models import ExtractionResult


class KnowledgeGraph:
    """
    Neo4j Knowledge Graph client for the Medical AI system.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password123",
    ) -> None:
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
        print(f"[KnowledgeGraph] Connected to Neo4j at {uri}")

    def close(self) -> None:
        self._driver.close()

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def create_indexes(self) -> None:
        """Create uniqueness constraints for all medical node types."""
        constraints = [
            ("Disease",          "disease_id"),
            ("Medication",       "drug_id"),
            ("Symptom",          "symptom_id"),
            ("MedicalProcedure", "procedure_id"),
            ("MedicalEntry",     "entry_id"),
        ]
        with self._driver.session() as session:
            for label, prop in constraints:
                cypher = (
                    f"CREATE CONSTRAINT IF NOT EXISTS "
                    f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
                )
                session.run(cypher)
        print("[KnowledgeGraph] Medical schema constraints/indexes created.")

    def clear_graph(self) -> None:
        """Delete all nodes and relationships (dev/reset only)."""
        with self._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("[KnowledgeGraph] Graph cleared.")

    # ------------------------------------------------------------------
    # Node creation (MERGE = idempotent)
    # ------------------------------------------------------------------

    def _merge_nodes(self, session, entities: ExtractionResult) -> None:

        # Disease nodes
        for d in entities.diseases:
            session.run(
                """MERGE (n:Disease {disease_id: $disease_id})
                   SET n.name = $name,
                       n.category = $category,
                       n.doc_id = $doc_id,
                       n.icd_codes = $icd_codes,
                       n.symptoms = $symptoms,
                       n.causes = $causes,
                       n.diagnostic_tests = $diagnostic_tests""",
                disease_id=d.disease_id, name=d.name, category=d.category,
                doc_id=d.doc_id, icd_codes=d.icd_codes,
                symptoms=d.symptoms, causes=d.causes,
                diagnostic_tests=d.diagnostic_tests,
            )

        # Medication nodes
        for m in entities.medications:
            session.run(
                """MERGE (n:Medication {drug_id: $drug_id})
                   SET n.name = $name,
                       n.generic_name = $generic_name,
                       n.brand_names = $brand_names,
                       n.drug_class = $drug_class,
                       n.doc_id = $doc_id,
                       n.indications = $indications,
                       n.side_effects = $side_effects,
                       n.contraindications = $contraindications""",
                drug_id=m.drug_id, name=m.name, generic_name=m.generic_name,
                brand_names=m.brand_names, drug_class=m.drug_class,
                doc_id=m.doc_id, indications=m.indications,
                side_effects=m.side_effects, contraindications=m.contraindications,
            )

        # Symptom nodes
        for s in entities.symptoms:
            session.run(
                """MERGE (n:Symptom {symptom_id: $symptom_id})
                   SET n.name = $name,
                       n.affected_body_part = $affected_body_part,
                       n.severity_levels = $severity_levels""",
                symptom_id=s.symptom_id, name=s.name,
                affected_body_part=s.affected_body_part,
                severity_levels=s.severity_levels,
            )

        # MedicalProcedure nodes
        for p in entities.procedures:
            session.run(
                """MERGE (n:MedicalProcedure {procedure_id: $procedure_id})
                   SET n.name = $name,
                       n.procedure_type = $procedure_type,
                       n.doc_id = $doc_id,
                       n.purpose = $purpose,
                       n.risks = $risks,
                       n.preparation = $preparation""",
                procedure_id=p.procedure_id, name=p.name,
                procedure_type=p.procedure_type, doc_id=p.doc_id,
                purpose=p.purpose, risks=p.risks, preparation=p.preparation,
            )

        # MedicalEntry nodes (document anchors)
        for e in entities.entries:
            session.run(
                """MERGE (n:MedicalEntry {entry_id: $entry_id})
                   SET n.title = $title,
                       n.entry_type = $entry_type,
                       n.doc_id = $doc_id,
                       n.related_entries = $related_entries""",
                entry_id=e.entry_id, title=e.title,
                entry_type=e.entry_type, doc_id=e.doc_id,
                related_entries=e.related_entries,
            )

    # ------------------------------------------------------------------
    # Relationship creation
    # ------------------------------------------------------------------

    # Maps node type → (label, id_property)
    _LABEL_ID_MAP = {
        "Disease":          ("Disease",          "disease_id"),
        "Medication":       ("Medication",       "drug_id"),
        "Symptom":          ("Symptom",          "symptom_id"),
        "MedicalProcedure": ("MedicalProcedure", "procedure_id"),
        "MedicalEntry":     ("MedicalEntry",     "entry_id"),
    }

    def _merge_relationships(self, session, entities: ExtractionResult) -> None:
        for rel in entities.relationships:
            src_info = self._LABEL_ID_MAP.get(rel.source_type)
            tgt_info = self._LABEL_ID_MAP.get(rel.target_type)

            if not src_info or not tgt_info:
                continue  # Skip unknown types

            src_label, src_prop = src_info
            tgt_label, tgt_prop = tgt_info

            cypher = (
                f"MATCH (src:{src_label}), (tgt:{tgt_label}) "
                f"WHERE src.{src_prop} = $source_id AND tgt.{tgt_prop} = $target_id "
                f"MERGE (src)-[r:{rel.relation_type}]->(tgt) "
                f"SET r.notes = $notes"
            )
            try:
                session.run(
                    cypher,
                    source_id=rel.source_id,
                    target_id=rel.target_id,
                    notes=rel.notes,
                )
            except Exception:
                pass  # Silently skip if nodes don't exist

    # ------------------------------------------------------------------
    # Populate
    # ------------------------------------------------------------------

    def populate(self, entities: ExtractionResult) -> dict:
        """
        Insert all medical entities and relationships into Neo4j.
        Returns counts of what was written.
        """
        with self._driver.session() as session:
            self._merge_nodes(session, entities)
            self._merge_relationships(session, entities)

        counts = self.get_node_counts()
        print(f"[KnowledgeGraph] Populated medical graph. Node counts: {counts}")
        return counts

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def run_cypher(self, query: str, params: Optional[dict] = None) -> list[dict]:
        """Execute a Cypher query and return results as list of dicts."""
        with self._driver.session() as session:
            result = session.run(query, **(params or {}))
            return [dict(record) for record in result]

    def get_node_counts(self) -> dict:
        """Return count of each medical node label."""
        labels = ["Disease", "Medication", "Symptom", "MedicalProcedure", "MedicalEntry"]
        counts = {}
        with self._driver.session() as session:
            for label in labels:
                result = session.run(f"MATCH (n:{label}) RETURN count(n) AS cnt")
                counts[label] = result.single()["cnt"]
            # Total relationships
            result = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt")
            counts["Relationships"] = result.single()["cnt"]
        return counts

    def get_schema(self) -> str:
        """
        Return a text description of the medical graph schema for LLM prompting
        (used by GraphRetriever to generate Cypher queries from natural language).
        """
        counts = self.get_node_counts()
        return f"""Neo4j Medical Knowledge Graph Schema (Gale Encyclopedia of Medicine):

Node Labels:
  - Disease          (disease_id, name, category, icd_codes[], symptoms[], causes[], diagnostic_tests[], doc_id)  [{counts.get('Disease', 0)} nodes]
  - Medication       (drug_id, name, generic_name, brand_names[], drug_class, indications[], side_effects[], contraindications[], doc_id)  [{counts.get('Medication', 0)} nodes]
  - Symptom          (symptom_id, name, affected_body_part, severity_levels[])  [{counts.get('Symptom', 0)} nodes]
  - MedicalProcedure (procedure_id, name, procedure_type, purpose[], risks[], preparation[], doc_id)  [{counts.get('MedicalProcedure', 0)} nodes]
  - MedicalEntry     (entry_id, title, entry_type, related_entries[], doc_id)  [{counts.get('MedicalEntry', 0)} nodes]

Relationship Types:
  - (Disease)-[:HAS_SYMPTOM]->(Symptom)
  - (Disease)-[:TREATED_BY]->(Medication)
  - (Disease)-[:TREATED_BY]->(MedicalProcedure)
  - (Disease)-[:DIAGNOSED_BY]->(MedicalProcedure)
  - (Disease)-[:ALTERNATIVE_TREATMENT]->(MedicalProcedure)
  - (Medication)-[:CAUSES_SIDE_EFFECT]->(Symptom)
  - (Medication)-[:CONTRAINDICATED_WITH]->(Disease)
  - (Medication)-[:CONTRAINDICATED_WITH]->(Medication)
  - (Medication)-[:INTERACTS_WITH]->(Medication)
  - (any)-[:RELATED_TO]->(any)

Total relationships: {counts.get('Relationships', 0)}
"""
