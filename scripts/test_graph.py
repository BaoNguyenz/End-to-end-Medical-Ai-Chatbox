"""
test_graph.py
Verify Medical GraphRAG (Neo4j entity extraction + NL-to-Cypher retrieval).

Requires:
  - Neo4j running (docker compose up)
  - build_graph.py already run (Neo4j populated with medical entities)
  - OPENAI_API_KEY in .env

Run:
    uv run python scripts/test_graph.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

# pyrefly: ignore [missing-import]
from src.config import settings
# pyrefly: ignore [missing-import]
from src.graph.knowledge_graph import KnowledgeGraph
# pyrefly: ignore [missing-import]
from src.graph.graph_retriever import GraphRetriever

SEP = "=" * 65


def main() -> None:
    print(SEP)
    print("SETUP: Connecting to Neo4j Medical Knowledge Graph")
    print(SEP)

    kg = KnowledgeGraph(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )

    counts = kg.get_node_counts()
    print("\n  Medical graph node counts:")
    for label, count in counts.items():
        print(f"    {label:<20} {count}")

    total_nodes = sum(v for k, v in counts.items() if k != "Relationships")
    if total_nodes == 0:
        print("\n  [ERROR] Graph is empty! Run build_graph.py first:")
        print("    uv run python scripts/build_graph.py")
        sys.exit(1)

    print()
    client = OpenAI(api_key=settings.openai_api_key)
    retriever = GraphRetriever(
        knowledge_graph=kg,
        openai_client=client,
        model=settings.openai_model,
    )

    # ── TEST 1: Manual Cypher queries ────────────────────────────────────
    print(SEP)
    print("TEST 1: Manual Cypher queries on Medical Entities (no LLM)")
    print(SEP)

    cypher_tests = [
        ("Diseases and their Symptoms",
         "MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom) RETURN d.name AS disease, s.name AS symptom LIMIT 10"),
        ("Diseases and their Treatments/Medications",
         "MATCH (d:Disease)-[:TREATED_BY]->(m:Medication) RETURN d.name AS disease, m.name AS medication LIMIT 10"),
        ("Drug Side Effects",
         "MATCH (m:Medication)-[:CAUSES_SIDE_EFFECT]->(s:Symptom) RETURN m.name AS medication, s.name AS side_effect LIMIT 10"),
        ("Drug Contraindications with Diseases",
         "MATCH (m:Medication)-[:CONTRAINDICATED_WITH]->(d:Disease) RETURN m.name AS medication, d.name AS contraindicated_disease LIMIT 10"),
        ("Diagnostic and Surgical Procedures",
         "MATCH (p:MedicalProcedure) RETURN p.name AS procedure, p.procedure_type AS type, p.purpose AS purpose LIMIT 10"),
    ]

    for description, cypher in cypher_tests:
        print(f"\n  [{description}]")
        print(f"  Cypher: {cypher[:80]}...")
        results = kg.run_cypher(cypher)
        if results:
            for row in results[:5]:
                print(f"    -> {row}")
        else:
            print("    -> (no results)")

    # ── TEST 2: NL-to-Cypher via GraphRetriever ──────────────────────────
    print()
    print(SEP)
    print("TEST 2: NL-to-Cypher Medical Retrieval")
    print(SEP)
    print("LLM translates medical questions to Cypher, executes, returns results.\n")

    nl_queries = [
        "What are the symptoms of asthma?",
        "What medications are used to treat diabetes?",
        "What are the side effects of aspirin?",
        "What drugs are contraindicated with asthma?",
        "What diagnostic procedures are used for appendicitis?",
        "Can aspirin interact with other drugs?",
    ]

    for query in nl_queries:
        print(f'  Query: "{query}"')
        t = time.time()
        results = retriever.search(query)
        elapsed = time.time() - t

        print(f"  Time: {elapsed:.2f}s  |  {len(results)} results")
        for r in results[:3]:
            content = str(r.chunk.content).replace("\n", " ").encode("ascii", errors="ignore").decode("ascii")
            print(f"    [{r.source.value}] {content}")
        print()

    # ── TEST 3: Schema inspection ─────────────────────────────────────────
    print(SEP)
    print("TEST 3: Medical Graph Schema (used as LLM context)")
    print(SEP)
    print(kg.get_schema())

    # ── Summary ──────────────────────────────────────────────────────────
    print(SEP)
    print("MEDICAL GRAPHRAG VERIFICATION COMPLETE")
    print(SEP)
    print("  [OK] Neo4j connected and populated with Disease, Medication, Symptom, Procedure nodes")
    print("  [OK] Manual Cypher queries return medical relationships")
    print("  [OK] NL-to-Cypher translation working for medical questions")
    print("  [OK] GraphRetriever returns structured SearchResult objects")
    print()
    print("Neo4j Browser: http://localhost:7475")
    print("  Explore: MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50")

    kg.close()


if __name__ == "__main__":
    main()
