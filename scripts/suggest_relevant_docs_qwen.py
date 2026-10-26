"""Suggest reviewable document labels for unlabeled retrieval benchmark questions.

Run after scripts/start_local_qwen.ps1. This script never edits the benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "Data/benchmarks/medical_benchmark_with_ground_truth.json"
CORPUS = ROOT / "Data/markdown_output"
OUTPUT = ROOT / "output/relevant_doc_suggestions_56.json"
TABLE = ROOT / "output/relevant_doc_suggestions_56.tsv"
MODEL = "Qwen/Qwen3-8B-GGUF:Q4_K_M"
URL = "http://127.0.0.1:11435/v1/chat/completions"


def snippet(text: str, query: str, limit: int = 900) -> str:
    """Show the opening and a question-term match from the source article."""
    clean = re.sub(r"\s+", " ", text)
    terms = [t.lower() for t in re.findall(r"[A-Za-z]{5,}", query)]
    terms = [t for t in terms if t not in {"what", "which", "their", "these", "about", "after", "versus"}]
    lower = clean.lower()
    positions = [lower.find(t, 250) for t in terms]
    positions = [p for p in positions if p >= 0]
    prefix = clean[: limit // 2]
    if positions:
        pos = min(positions)
        return prefix + " ... " + clean[max(0, pos - 120) : pos + limit // 2]
    return clean[:limit]


def parse_json_response(content: str) -> dict:
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
    return json.loads(content)


def sanity_check(result: dict) -> dict:
    """Filter redirect stubs and flag proposals that need closer human review."""
    proposed = result["suggested_relevant_doc_ids"]
    removed = [doc_id for doc_id in result.get("raw_qwen_doc_ids", proposed) if "_see_" in doc_id]
    if removed:
        result["raw_qwen_doc_ids"] = result.get("raw_qwen_doc_ids", proposed)
        result["suggested_relevant_doc_ids"] = [doc_id for doc_id in proposed if doc_id not in removed]
        result["selected_sources"] = [
            source for source in result["selected_sources"] if source["doc_id"] not in removed
        ]
    flags = []
    if removed:
        flags.append("redirect_stub_removed")
    if not result["suggested_relevant_doc_ids"]:
        flags.append("no_supported_source")
    if len(result["suggested_relevant_doc_ids"]) >= 3:
        flags.append("many_sources_check_minimality")
    if result["ground_truth_status"] != "verified":
        flags.append("ground_truth_source_missing")
    if any("mark" in doc_id for doc_id in result["suggested_relevant_doc_ids"]):
        flags.append("corrupt_source_filename_check_content")
    if result["confidence"] != "high":
        flags.append("low_model_confidence")
    if "allergic conjunctivitis" in result["query"].lower() and any(
        "Adenovirus" in doc_id for doc_id in result["suggested_relevant_doc_ids"]
    ):
        flags.append("possible_allergic_vs_infectious_topic_mismatch")
    result["review_flags"] = flags
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="Optional number of questions for a trial")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    questions = [
        q for q in benchmark["questions"]
        if not q.get("relevant_doc_ids") and not q["is_emergency"] and not q["expect_rejection"]
    ]
    if len(questions) != 56:
        raise RuntimeError(f"Expected 56 missing retrieval labels; found {len(questions)}")
    if args.limit:
        questions = questions[: args.limit]

    paths = sorted(CORPUS.glob("*.md"))
    texts = [p.read_text(encoding="utf-8", errors="replace") for p in paths]
    ids = [p.stem for p in paths]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=50000)
    matrix = vectorizer.fit_transform([
        p.stem.replace("_", " ") * 5 + " " + t[:14000] for p, t in zip(paths, texts)
    ])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    if args.output.exists():
        old = json.loads(args.output.read_text(encoding="utf-8"))
        results = old.get("suggestions", [])
    results = [sanity_check(r) for r in results]
    done = {r["id"] for r in results}

    for q in questions:
        if q["id"] in done:
            continue
        scores = cosine_similarity(vectorizer.transform([q["query"] + " " + q["ground_truth"]]), matrix)[0]
        ranked = sorted(range(len(paths)), key=lambda i: -scores[i])[:8]
        candidates = [
            {"doc_id": ids[i], "tfidf_score": round(float(scores[i]), 4),
             "excerpt": snippet(texts[i], q["query"])} for i in ranked
        ]
        system = (
            "You are assigning STRICT, MINIMAL gold relevant_doc_ids for a medical QA retrieval benchmark. "
            "Select source articles from candidates that DIRECTLY answer the question, based on article CONTENT. "
            "Usually one source is sufficient; select a second only when a distinct part of the question needs it. "
            "Never add background, loosely related, tangential, or redundant articles. "
            "Never select a cross-reference stub. Some articles have mixed-up content under a correct title; "
            "a matching title alone is NOT enough. If an article's shown content is corrupt or insufficient, "
            "exclude it and explain. Return [] when no candidate directly supports the answer. "
            "Do not invent IDs or medical facts. Return JSON with keys relevant_doc_ids (array), "
            "review_note (short English explanation), and confidence (high/medium/low)."
        )
        payload = {
            "model": MODEL,
            "temperature": 0,
            "max_tokens": 350,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps({
                    "id": q["id"], "query": q["query"], "reference_answer": q["ground_truth"],
                    "candidates": candidates,
                }, ensure_ascii=False)},
            ],
        }
        response = requests.post(URL, json=payload, timeout=180)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        parsed = parse_json_response(content)
        allowed = {c["doc_id"] for c in candidates}
        proposed = list(dict.fromkeys(parsed.get("relevant_doc_ids", [])))
        invalid = [doc_id for doc_id in proposed if doc_id not in allowed]
        if invalid:
            raise ValueError(f"{q['id']}: Qwen returned IDs outside candidates: {invalid}")
        selected = [c for c in candidates if c["doc_id"] in proposed]
        results.append(sanity_check({
            "id": q["id"], "category": q["category"], "query": q["query"],
            "ground_truth_status": q.get("ground_truth_status", "verified"),
            "suggested_relevant_doc_ids": proposed,
            "confidence": parsed.get("confidence", "low"),
            "review_note": parsed.get("review_note", ""),
            "selected_sources": [
                {"doc_id": c["doc_id"], "source": str(CORPUS.relative_to(ROOT) / (c["doc_id"] + ".md")),
                 "excerpt": c["excerpt"]} for c in selected
            ],
            "candidate_doc_ids": [c["doc_id"] for c in candidates],
            "review_status": "pending",
        }))
        document = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "benchmark": str(BENCHMARK.relative_to(ROOT)),
            "model": MODEL,
            "scope": "Unlabeled retrieval questions only; source benchmark unchanged",
            "target_count": len(questions), "completed_count": len(results),
            "suggestions": results,
        }
        args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{len(results):02d}/{len(questions)} {q['id']}: {', '.join(proposed) or '(none)'}", flush=True)

    table_path = TABLE if args.output == OUTPUT else args.output.with_suffix(".tsv")
    document = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": str(BENCHMARK.relative_to(ROOT)),
        "model": MODEL,
        "scope": "Unlabeled retrieval questions only; source benchmark unchanged",
        "target_count": len(questions), "completed_count": len(results),
        "suggestions": results,
    }
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with table_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["id", "category", "query", "suggested_relevant_doc_ids", "confidence", "review_flags", "review_note", "ground_truth_status", "review_status"])
        for r in results:
            writer.writerow([r["id"], r["category"], r["query"], "; ".join(r["suggested_relevant_doc_ids"]), r["confidence"], "; ".join(r["review_flags"]), r["review_note"], r["ground_truth_status"], r["review_status"]])
    print(f"JSON: {args.output}\nTable: {table_path}")


if __name__ == "__main__":
    main()
