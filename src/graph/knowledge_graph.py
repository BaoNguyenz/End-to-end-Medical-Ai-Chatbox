"""
knowledge_graph.py
Neo4j graph database operations for the Enterprise RAG system.

Handles:
  - Connection management
  - Schema setup (indexes + constraints)
  - Node and relationship creation via MERGE (idempotent)
  - Arbitrary Cypher query execution
  - Schema introspection for LLM context
"""

from __future__ import annotations

from typing import Optional

from neo4j import GraphDatabase, Driver

from src.graph.entity_models import ExtractionResult


class KnowledgeGraph:
    """
    Manages the Neo4j knowledge graph for TechDocs entities.

    All write operations use MERGE to be fully idempotent — safe to re-run.
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
        # Verify connectivity
        self._driver.verify_connectivity()
        print(f"[KnowledgeGraph] Connected to Neo4j at {uri}")

    def close(self) -> None:
        self._driver.close()

    # ------------------------------------------------------------------
    # Schema setup
    # ------------------------------------------------------------------

    def create_indexes(self) -> None:
        """Create uniqueness constraints and indexes for fast lookup."""
        constraints = [
            ("Policy",       "policy_id"),
            ("Product",      "product_id"),
            ("TechnicalDoc", "doc_id"),
            ("Stakeholder",  "name"),
            ("Regulation",   "name"),
        ]
        with self._driver.session() as session:
            for label, prop in constraints:
                cypher = (
                    f"CREATE CONSTRAINT IF NOT EXISTS "
                    f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
                )
                session.run(cypher)
        print("[KnowledgeGraph] Constraints/indexes created.")

    def clear_graph(self) -> None:
        """Delete all nodes and relationships (dev/reset only)."""
        with self._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("[KnowledgeGraph] Graph cleared.")

    # ------------------------------------------------------------------
    # Node creation (MERGE = idempotent)
    # ------------------------------------------------------------------

    def _merge_nodes(self, session, entities: ExtractionResult) -> None:
        # Policies
        for p in entities.policies:
            session.run(
                """MERGE (n:Policy {policy_id: $policy_id})
                   SET n.name = $name, n.owner = $owner,
                       n.effective_date = $effective_date,
                       n.review_date = $review_date,
                       n.doc_id = $doc_id,
                       n.regulations = $regulations""",
                policy_id=p.policy_id, name=p.name, owner=p.owner,
                effective_date=p.effective_date, review_date=p.review_date,
                doc_id=p.doc_id, regulations=p.regulations,
            )

        # Stakeholders
        for s in entities.stakeholders:
            session.run(
                """MERGE (n:Stakeholder {name: $name})
                   SET n.role = $role,
                       n.responsibilities = $responsibilities""",
                name=s.name, role=s.role,
                responsibilities=s.responsibilities,
            )

        # Products
        for p in entities.products:
            session.run(
                """MERGE (n:Product {product_id: $product_id})
                   SET n.name = $name, n.category = $category,
                       n.version = $version, n.doc_id = $doc_id,
                       n.features = $features""",
                product_id=p.product_id, name=p.name,
                category=p.category, version=p.version,
                doc_id=p.doc_id, features=p.features,
            )

        # Regulations
        for r in entities.regulations:
            session.run(
                """MERGE (n:Regulation {name: $name})
                   SET n.articles = $articles""",
                name=r.name, articles=r.articles,
            )

        # TechnicalDocs
        for t in entities.technical_docs:
            session.run(
                """MERGE (n:TechnicalDoc {doc_id: $doc_id})
                   SET n.title = $title, n.version = $version,
                       n.error_codes = $error_codes,
                       n.technologies = $technologies""",
                doc_id=t.doc_id, title=t.title, version=t.version,
                error_codes=t.error_codes, technologies=t.technologies,
            )

    # ------------------------------------------------------------------
    # Relationship creation
    # ------------------------------------------------------------------

    # Map each (source_type, target_type) combination to the right MERGE Cypher
    _REL_TEMPLATE = """
        MATCH (src:{src_label}), (tgt:{tgt_label})
        WHERE {src_key} = $source_id AND {tgt_key} = $target_id
        MERGE (src)-[r:{rel_type}]->(tgt)
    """

    _LABEL_ID_MAP = {
        "Policy":       ("Policy",       "src.policy_id"),
        "Stakeholder":  ("Stakeholder",  "src.name"),
        "Product":      ("Product",      "src.product_id"),
        "Regulation":   ("Regulation",   "src.name"),
        "TechnicalDoc": ("TechnicalDoc", "src.doc_id"),
    }

    _TARGET_ID_MAP = {
        "Policy":       "tgt.policy_id",
        "Stakeholder":  "tgt.name",
        "Product":      "tgt.product_id",
        "Regulation":   "tgt.name",
        "TechnicalDoc": "tgt.doc_id",
    }

    def _merge_relationships(self, session, entities: ExtractionResult) -> None:
        for rel in entities.relationships:
            src_label, src_key = self._LABEL_ID_MAP.get(rel.source_type, (rel.source_type, "src.name"))
            tgt_key = self._TARGET_ID_MAP.get(rel.target_type, "tgt.name")

            cypher = (
                f"MATCH (src:{src_label}), (tgt:{rel.target_type}) "
                f"WHERE {src_key} = $source_id AND {tgt_key} = $target_id "
                f"MERGE (src)-[r:{rel.relation_type}]->(tgt)"
            )
            try:
                session.run(cypher, source_id=rel.source_id, target_id=rel.target_id)
            except Exception as e:
                # Silently skip relationship if nodes don't exist
                pass

    # ------------------------------------------------------------------
    # Populate
    # ------------------------------------------------------------------

    def populate(self, entities: ExtractionResult) -> dict:
        """
        Insert all entities and relationships into Neo4j.
        Returns counts of what was written.
        """
        with self._driver.session() as session:
            self._merge_nodes(session, entities)
            self._merge_relationships(session, entities)

        # Count what's in the graph
        counts = self.get_node_counts()
        print(f"[KnowledgeGraph] Populated. Node counts: {counts}")
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
        """Return count of each node label."""
        labels = ["Policy", "Stakeholder", "Product", "Regulation", "TechnicalDoc"]
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
        Return a text description of the graph schema for LLM prompting.
        """
        counts = self.get_node_counts()
        return f"""Neo4j Graph Schema for TechDocs Inc.:

Node Labels:
  - Policy       (policy_id, name, owner, effective_date, regulations[])   [{counts.get('Policy', 0)} nodes]
  - Stakeholder  (name, role, responsibilities[])                          [{counts.get('Stakeholder', 0)} nodes]
  - Product      (product_id, name, category, version, features[])         [{counts.get('Product', 0)} nodes]
  - Regulation   (name, articles[])                                        [{counts.get('Regulation', 0)} nodes]
  - TechnicalDoc (doc_id, title, version, error_codes[], technologies[])   [{counts.get('TechnicalDoc', 0)} nodes]

Relationship Types:
  - (Policy)-[:OWNED_BY]->(Stakeholder)
  - (Policy)-[:COMPLIES_WITH]->(Regulation)
  - (Policy)-[:REFERENCES]->(Policy)
  - (Stakeholder)-[:RESPONSIBLE_FOR]->(Policy)
  - (TechnicalDoc)-[:OWNED_BY]->(Stakeholder)
  - (TechnicalDoc)-[:REFERENCES]->(TechnicalDoc)
  - (any)-[:RELATES_TO]->(any)

Total relationships: {counts.get('Relationships', 0)}
"""
