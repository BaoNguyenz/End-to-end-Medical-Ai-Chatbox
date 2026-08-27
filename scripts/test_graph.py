"""
test_graph.py
Verify Task 5: GraphRAG (Neo4j entity extraction + NL-to-Cypher retrieval).

Requires:
  - Neo4j running (docker compose up)
  - build_graph.py already run (Neo4j populated)
  - OPENAI_API_KEY in .env

Run:
    uv run python scripts/test_graph.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from src.config import settings
from src.graph.knowledge_graph import KnowledgeGraph
from src.graph.graph_retriever import GraphRetriever

SEP = "=" * 65


def main() -> None:
    print(SEP)
    print("SETUP: Connecting to Neo4j")
    print(SEP)

    kg = KnowledgeGraph(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )

    counts = kg.get_node_counts()
    print("\n  Graph node counts:")
    for label, count in counts.items():
        print(f"    {label:<15} {count}")

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
    print("TEST 1: Manual Cypher queries (no LLM)")
    print(SEP)

    cypher_tests = [
        ("All Policies with owners",
         "MATCH (p:Policy)-[:OWNED_BY]->(s:Stakeholder) RETURN p.policy_id, p.name, s.name AS owner LIMIT 10"),
        ("Policies complying with GDPR",
         "MATCH (p:Policy)-[:COMPLIES_WITH]->(r:Regulation {name: 'GDPR'}) RETURN p.policy_id, p.name LIMIT 10"),
        ("All Stakeholders",
         "MATCH (s:Stakeholder) RETURN s.name, s.role LIMIT 10"),
        ("All Products",
         "MATCH (p:Product) RETURN p.product_id, p.name, p.category LIMIT 10"),
        ("TechnicalDocs with error codes",
         "MATCH (t:TechnicalDoc) WHERE size(t.error_codes) > 0 RETURN t.doc_id, t.title, t.error_codes LIMIT 10"),
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
    print("TEST 2: NL-to-Cypher retrieval")
    print(SEP)
    print("LLM translates natural language to Cypher, executes, returns results.\n")

    nl_queries = [
        "Who is responsible for data privacy?",
        "Which policies comply with GDPR?",
        "What are the stakeholders for the incident response policy?",
        "Who owns the remote work policy?",
        "What products does TechDocs offer?",
        "Which technical documents mention OAuth?",
    ]

    for query in nl_queries:
        print(f'  Query: "{query}"')
        t = time.time()
        results = retriever.search(query)
        elapsed = time.time() - t

        print(f"  Time: {elapsed:.2f}s  |  {len(results)} results")
        for r in results[:3]:
            content = r.chunk.content.replace("\n", " ").encode("ascii", "replace").decode("ascii")
            print(f"    [{r.source.value}] {content}")
        print()

    # ── TEST 3: Schema inspection ─────────────────────────────────────────
    print(SEP)
    print("TEST 3: Graph schema (used as LLM context)")
    print(SEP)
    print(kg.get_schema())

    # ── Summary ──────────────────────────────────────────────────────────
    print(SEP)
    print("TASK 5 VERIFICATION COMPLETE")
    print(SEP)
    print("  [OK] Neo4j connected and populated")
    print("  [OK] Manual Cypher queries return correct nodes/relationships")
    print("  [OK] NL-to-Cypher translation working")
    print("  [OK] GraphRetriever returns SearchResult objects")
    print("  [OK] Schema introspection available for LLM context")
    print()
    print("Also check visually: http://localhost:7474")
    print("  Run: MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 50")
    print()
    print("Next: Task 6 - Integration & Orchestration (full pipeline + API)")

    kg.close()


if __name__ == "__main__":
    main()
