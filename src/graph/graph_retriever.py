"""
graph_retriever.py
Retrieves information from the Neo4j knowledge graph using:
  1. NL-to-Cypher: LLM translates natural language queries into Cypher.
  2. Fallback: keyword search on node properties if Cypher fails.

Results are returned as structured GraphHit objects. Their provenance is
resolved to canonical Qdrant chunks before passage reranking.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

from src.graph.knowledge_graph import KnowledgeGraph


@dataclass
class GraphHit:
    """Structured graph fact plus provenance to canonical text chunks."""

    content: str
    evidence_chunk_ids: list[str] = field(default_factory=list)
    source_doc_ids: list[str] = field(default_factory=list)
    score: float = 0.0
    method: str = "cypher"
    record: dict = field(default_factory=dict)


_NL_TO_CYPHER_PROMPT = """You are an expert Neo4j Cypher query writer for a medical knowledge graph.

{schema}

Translate the following medical question into a valid Cypher query.
Rules:
- ONLY output the raw Cypher query. No explanation, no markdown, no backticks.
- ALWAYS use case-insensitive matching with toLower(...) CONTAINS '...' (e.g., WHERE toLower(d.name) CONTAINS 'asthma').
- ALWAYS use valid relationship types from the schema: HAS_SYMPTOM, TREATED_BY,
  DIAGNOSED_BY, CAUSES_SIDE_EFFECT, CONTRAINDICATED_WITH, INTERACTS_WITH,
  ALTERNATIVE_TREATMENT, RELATED_TO.
- Use MATCH, WHERE, RETURN. Keep it simple and correct.
- Always add LIMIT 10.
- Return meaningful string fields (e.g. d.name, s.name, m.name, m.drug_class).
- Bind matched medical relationships as `r` whenever possible.
- For relationship queries ALWAYS return r.evidence_chunk_ids AS evidence_chunk_ids
  and r.source_doc_ids AS source_doc_ids.

Examples:
  Q: "What are the symptoms of asthma?"
  A: MATCH (d:Disease)-[r:HAS_SYMPTOM]->(s:Symptom) WHERE toLower(d.name) CONTAINS 'asthma' RETURN d.name AS disease, s.name AS symptom, r.evidence_chunk_ids AS evidence_chunk_ids, r.source_doc_ids AS source_doc_ids LIMIT 10

  Q: "Which medications treat diabetes?"
  A: MATCH (d:Disease)-[r:TREATED_BY]->(m:Medication) WHERE toLower(d.name) CONTAINS 'diabetes' RETURN d.name AS disease, m.name AS medication, r.evidence_chunk_ids AS evidence_chunk_ids, r.source_doc_ids AS source_doc_ids LIMIT 10

  Q: "What drugs interact with aspirin?"
  A: MATCH (m:Medication)-[r:INTERACTS_WITH]->(m2:Medication) WHERE toLower(m.name) CONTAINS 'aspirin' RETURN m.name AS drug, m2.name AS interacts_with, r.evidence_chunk_ids AS evidence_chunk_ids, r.source_doc_ids AS source_doc_ids LIMIT 10

  Q: "What medications are contraindicated with asthma?"
  A: MATCH (m:Medication)-[r:CONTRAINDICATED_WITH]->(d:Disease) WHERE toLower(d.name) CONTAINS 'asthma' RETURN m.name AS drug, d.name AS disease, r.evidence_chunk_ids AS evidence_chunk_ids, r.source_doc_ids AS source_doc_ids LIMIT 10

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
      4. Return structured graph facts with evidence chunk IDs
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
        # Extract main nouns/keywords from query
        words = [w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", query)
                 if w.lower() not in {"what", "are", "the", "and", "for", "with", "how", "does", "which", "used", "treat"}]
        search_term = words[0] if words else query.lower()

        cypher = """
        MATCH (n)
        WHERE toLower(n.name) CONTAINS $q
           OR toLower(coalesce(n.disease_id, '')) CONTAINS $q
           OR toLower(coalesce(n.drug_id, '')) CONTAINS $q
           OR any(r IN coalesce(n.symptoms, []) WHERE toLower(r) CONTAINS $q)
           OR any(f IN coalesce(n.indications, []) WHERE toLower(f) CONTAINS $q)
        OPTIONAL MATCH (c:Chunk)
        WHERE c.doc_id = coalesce(n.doc_id, '')
        RETURN labels(n)[0] AS type, n.name AS name,
               coalesce(n.disease_id, n.drug_id, n.procedure_id, n.entry_id, '') AS id,
               collect(DISTINCT c.chunk_id) AS evidence_chunk_ids,
               [coalesce(n.doc_id, '')] AS source_doc_ids
        LIMIT 10
        """
        return self.kg.run_cypher(cypher, {"q": search_term})

    # ------------------------------------------------------------------
    # Main search
    # ------------------------------------------------------------------

    def search(self, query: str) -> list[GraphHit]:
        """
        Search the knowledge graph using natural language.

        Attempts NL-to-Cypher first, falls back to keyword search on error.

        Returns:
            Structured graph facts with source chunk provenance.
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
                elif not any(record.get("evidence_chunk_ids") for record in results):
                    # A valid Text2Cypher answer may omit provenance. Retain its
                    # facts, but add node-level chunk provenance so the graph
                    # branch can still contribute real passages to reranking.
                    results.extend(self._keyword_search(query))
                    method = "cypher_with_keyword_provenance"
            else:
                print("[GraphRetriever] Invalid Cypher generated, using keyword fallback")
                results = self._keyword_search(query)
                method = "keyword_fallback"

        except Exception as e:
            print(f"[GraphRetriever] Cypher error ({e}), using keyword fallback")
            results = self._keyword_search(query)
            method = "keyword_fallback"

        # Step 2: Keep graph facts separate from text passages. Only the
        # evidence chunks referenced here will enter passage reranking.
        search_results: list[GraphHit] = []
        for i, record in enumerate(results):
            evidence_ids = record.get("evidence_chunk_ids") or []
            source_doc_ids = record.get("source_doc_ids") or []
            if isinstance(evidence_ids, str):
                evidence_ids = [evidence_ids]
            if isinstance(source_doc_ids, str):
                source_doc_ids = [source_doc_ids]
            fact_record = {
                key: value for key, value in record.items()
                if key not in {"evidence_chunk_ids", "source_doc_ids"}
            }
            search_results.append(GraphHit(
                content=_format_record_as_text(fact_record),
                evidence_chunk_ids=list(dict.fromkeys(evidence_ids)),
                source_doc_ids=list(dict.fromkeys(x for x in source_doc_ids if x)),
                score=1.0 - (i * 0.05),  # rank-based score
                method=method,
                record=record,
            ))

        print(f"[GraphRetriever] Returned {len(search_results)} graph results (method={method})")
        return search_results
