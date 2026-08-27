"""
graph_retriever.py
Retrieves information from the Neo4j knowledge graph using:
  1. NL-to-Cypher: LLM translates natural language queries into Cypher.
  2. Fallback: keyword search on node properties if Cypher fails.

Results are returned as SearchResult objects for compatibility with
the rest of the retrieval pipeline.
"""

from __future__ import annotations

import re
import time
from typing import Optional

from openai import OpenAI

from src.models import Chunk, SearchResult, SearchSource
from src.graph.knowledge_graph import KnowledgeGraph


_NL_TO_CYPHER_PROMPT = """You are an expert Neo4j Cypher query writer.

{schema}

Translate the following question into a valid Cypher query.
- ONLY output the raw Cypher query. No explanation, no markdown, no backticks.
- Use MATCH, WHERE, RETURN. Keep it simple and correct.
- Always add LIMIT 10 unless the question asks for all.
- Return meaningful string fields (names, IDs, descriptions).

Examples:
  Q: "Who owns the data privacy policy?"
  A: MATCH (p:Policy {{policy_id: 'POL-001'}})-[:OWNED_BY]->(s:Stakeholder) RETURN s.name, s.role LIMIT 10

  Q: "Which policies comply with GDPR?"
  A: MATCH (p:Policy)-[:COMPLIES_WITH]->(r:Regulation {{name: 'GDPR'}}) RETURN p.policy_id, p.name LIMIT 10

  Q: "What are the stakeholders for incident response?"
  A: MATCH (p:Policy)-[:OWNED_BY]->(s:Stakeholder) WHERE toLower(p.name) CONTAINS 'incident' RETURN p.name, s.name, s.role LIMIT 10

Question: {query}
Cypher:"""


def _is_valid_cypher(cypher: str) -> bool:
    """Basic validation: must contain MATCH and RETURN."""
    upper = cypher.upper()
    return "MATCH" in upper and "RETURN" in upper


def _format_record_as_text(record: dict) -> str:
    """Convert a Neo4j result record dict to a readable string."""
    parts = []
    for key, value in record.items():
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        parts.append(f"{key}: {value}")
    return " | ".join(parts)


class GraphRetriever:
    """
    Retrieves information from Neo4j using natural language queries.

    Flow:
      1. Ask LLM to translate query -> Cypher
      2. Execute Cypher against Neo4j
      3. If Cypher fails or returns nothing -> fallback keyword search
      4. Return results as SearchResult objects
    """

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        openai_client: OpenAI,
        model: str,
    ) -> None:
        self.kg = knowledge_graph
        self.client = openai_client
        self.model = model
        self._schema: Optional[str] = None  # lazy-loaded

    def _get_schema(self) -> str:
        if self._schema is None:
            self._schema = self.kg.get_schema()
        return self._schema

    # ------------------------------------------------------------------
    # NL to Cypher
    # ------------------------------------------------------------------

    def nl_to_cypher(self, query: str) -> str:
        """Ask the LLM to translate query to a Cypher statement."""
        schema = self._get_schema()
        prompt = _NL_TO_CYPHER_PROMPT.format(schema=schema, query=query)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        cypher = response.choices[0].message.content.strip()

        # Strip markdown fences if model added them
        cypher = re.sub(r"```(?:cypher)?", "", cypher, flags=re.IGNORECASE).strip("`").strip()
        return cypher

    # ------------------------------------------------------------------
    # Fallback keyword search
    # ------------------------------------------------------------------

    def _keyword_search(self, query: str) -> list[dict]:
        """Search node properties for keyword matches (no LLM needed)."""
        q = query.lower()
        cypher = """
        MATCH (n)
        WHERE toLower(n.name) CONTAINS $q
           OR toLower(coalesce(n.policy_id, '')) CONTAINS $q
           OR toLower(coalesce(n.doc_id, '')) CONTAINS $q
           OR any(r IN coalesce(n.regulations, []) WHERE toLower(r) CONTAINS $q)
           OR any(f IN coalesce(n.features, []) WHERE toLower(f) CONTAINS $q)
        RETURN labels(n)[0] AS type, n.name AS name,
               coalesce(n.policy_id, n.product_id, n.doc_id, '') AS id
        LIMIT 10
        """
        return self.kg.run_cypher(cypher, {"q": q})

    # ------------------------------------------------------------------
    # Main search
    # ------------------------------------------------------------------

    def search(self, query: str) -> list[SearchResult]:
        """
        Search the knowledge graph using natural language.

        Attempts NL-to-Cypher first, falls back to keyword search on error.

        Returns:
            List of SearchResult where each result wraps one graph record.
        """
        results: list[dict] = []
        method = "cypher"

        # Step 1: NL to Cypher
        try:
            t = time.time()
            cypher = self.nl_to_cypher(query)
            print(f"[GraphRetriever] Cypher ({time.time()-t:.1f}s): {cypher[:100]}")

            if _is_valid_cypher(cypher):
                results = self.kg.run_cypher(cypher)
                if not results:
                    print("[GraphRetriever] Cypher returned 0 results, trying keyword fallback")
                    results = self._keyword_search(query)
                    method = "keyword_fallback"
            else:
                print("[GraphRetriever] Invalid Cypher generated, using keyword fallback")
                results = self._keyword_search(query)
                method = "keyword_fallback"

        except Exception as e:
            print(f"[GraphRetriever] Cypher error ({e}), using keyword fallback")
            results = self._keyword_search(query)
            method = "keyword_fallback"

        # Step 2: Convert records to SearchResult
        search_results: list[SearchResult] = []
        for i, record in enumerate(results):
            text = _format_record_as_text(record)
            chunk = Chunk(
                content=text,
                chunk_id=f"graph_{method}_{i}",
                doc_id=f"graph:{method}",
                document_type="graph",
                chunk_index=i,
                metadata={"method": method, "record": record},
            )
            search_results.append(SearchResult(
                chunk=chunk,
                score=1.0 - (i * 0.05),  # rank-based score
                source=SearchSource.GRAPH,
            ))

        print(f"[GraphRetriever] Returned {len(search_results)} graph results (method={method})")
        return search_results
