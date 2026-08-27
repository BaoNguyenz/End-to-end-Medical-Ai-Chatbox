"""
evaluator.py
Logging and metrics for the RAG pipeline.

Metrics computed:
  - Context Relevance: avg cosine similarity between query and retrieved chunks
  - Answer Faithfulness: LLM judge — is the answer grounded in the context?
  - Latency breakdown per stage
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from src.models import RAGResponse


_FAITHFULNESS_PROMPT = """You are an evaluation judge for a RAG (Retrieval-Augmented Generation) system.

Context provided to the model:
{context}

Generated answer:
{answer}

Task: Score how faithful/grounded the answer is to the provided context.
- 1.0 = every claim in the answer is directly supported by the context
- 0.5 = most claims are supported, minor additions/inferences
- 0.0 = the answer contains significant information not in the context

Return ONLY a JSON object: {{"score": <float 0.0-1.0>, "reason": "<one sentence>"}}"""


class Evaluator:
    """
    Evaluates RAG pipeline responses and logs metrics.
    """

    def __init__(
        self,
        embedding_model: SentenceTransformer,
        openai_client: OpenAI,
        openai_model: str,
        log_dir: Optional[Path] = None,
    ) -> None:
        self.embedding_model = embedding_model
        self.client = openai_client
        self.model = openai_model
        self.log_dir = log_dir or Path("cache")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log: list[dict] = []

    # ------------------------------------------------------------------
    # Metric: Context Relevance
    # ------------------------------------------------------------------

    def context_relevance(self, query: str, response: RAGResponse) -> float:
        """
        Average cosine similarity between query embedding and each context chunk embedding.
        Range: [0, 1]. Higher = more relevant context retrieved.
        """
        if not response.sources:
            return 0.0

        texts = [query] + [r.chunk.content for r in response.sources]
        embeddings = self.embedding_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        query_emb = embeddings[0]
        chunk_embs = embeddings[1:]
        similarities = chunk_embs @ query_emb
        return float(np.mean(similarities))

    # ------------------------------------------------------------------
    # Metric: Answer Faithfulness
    # ------------------------------------------------------------------

    def answer_faithfulness(self, response: RAGResponse) -> tuple[float, str]:
        """
        Ask LLM to judge if the answer is grounded in the provided context.
        Returns (score 0-1, reason string).
        """
        context = "\n\n".join(
            f"[{i+1}] {r.chunk.doc_id}:\n{r.chunk.content}"
            for i, r in enumerate(response.sources[:5])
        )
        prompt = _FAITHFULNESS_PROMPT.format(
            context=context,
            answer=response.answer,
        )

        try:
            llm_resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
                response_format={"type": "json_object"},
            )
            data = json.loads(llm_resp.choices[0].message.content)
            return float(data.get("score", 0.5)), str(data.get("reason", ""))
        except Exception as e:
            return 0.5, f"eval error: {e}"

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    def evaluate(self, response: RAGResponse, run_faithfulness: bool = True) -> dict:
        """
        Compute all metrics for a RAGResponse.

        Returns:
            dict with keys: query, context_relevance, faithfulness_score,
                            faithfulness_reason, latency, num_sources
        """
        ctx_rel = self.context_relevance(response.query, response)

        faith_score, faith_reason = 0.5, "skipped"
        if run_faithfulness:
            faith_score, faith_reason = self.answer_faithfulness(response)

        record = {
            "query": response.query,
            "answer_preview": response.answer[:150],
            "context_relevance": round(ctx_rel, 4),
            "faithfulness_score": round(faith_score, 4),
            "faithfulness_reason": faith_reason,
            "num_sources": len(response.sources),
            "latency": response.latency,
            "metadata": response.metadata,
        }
        self._log.append(record)
        return record

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate a markdown evaluation report from logged results.
        """
        if not self._log:
            return "No evaluation data logged yet."

        lines = [
            "# RAG System Evaluation Report",
            "",
            f"**Total queries evaluated:** {len(self._log)}",
            "",
            "## Summary Metrics",
            "",
        ]

        avg_ctx = np.mean([r["context_relevance"] for r in self._log])
        avg_faith = np.mean([r["faithfulness_score"] for r in self._log])
        avg_latency = np.mean([r["latency"].get("total", 0) for r in self._log])

        lines += [
            f"| Metric | Average |",
            f"|--------|---------|",
            f"| Context Relevance | {avg_ctx:.4f} |",
            f"| Answer Faithfulness | {avg_faith:.4f} |",
            f"| Total Latency (s) | {avg_latency:.2f} |",
            "",
            "## Per-Query Results",
            "",
            "| # | Query | Ctx Relevance | Faithfulness | Latency |",
            "|---|-------|--------------|-------------|---------|",
        ]

        for i, r in enumerate(self._log):
            q = r["query"][:50].replace("|", "/")
            lat = r["latency"].get("total", 0)
            lines.append(
                f"| {i+1} | {q} | {r['context_relevance']:.4f} | "
                f"{r['faithfulness_score']:.4f} | {lat:.2f}s |"
            )

        lines += ["", "## Latency Breakdown (avg)", ""]
        stage_keys = ["query_classify", "retrieval", "graph_search", "post_retrieval", "generation"]
        for key in stage_keys:
            vals = [r["latency"].get(key, 0) for r in self._log if key in r["latency"]]
            if vals:
                lines.append(f"- **{key}**: {np.mean(vals):.3f}s avg")

        lines += ["", "## Detailed Results", ""]
        for i, r in enumerate(self._log):
            lines += [
                f"### Query {i+1}",
                f"**Q:** {r['query']}",
                f"**A:** {r['answer_preview']}...",
                f"- Context Relevance: {r['context_relevance']}",
                f"- Faithfulness: {r['faithfulness_score']} — {r['faithfulness_reason']}",
                f"- Sources: {r['num_sources']} chunks",
                "",
            ]

        report = "\n".join(lines)

        if output_path:
            output_path.write_text(report, encoding="utf-8")
            print(f"[Evaluator] Report saved to {output_path}")

        return report
