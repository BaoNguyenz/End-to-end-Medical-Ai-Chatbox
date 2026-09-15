"""
Generate Missing Ground Truths for 47 Template Questions
=========================================================
Upgrades the GaleMed Medical Benchmark from 58 to 105 complete gold standard
questions by replacing placeholder templates with medically precise ground truths.

Usage:
    python scripts/generate_missing_ground_truths.py
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BENCHMARK_PATH = Path("Data/benchmarks/medical_benchmark_with_ground_truth.json")
BACKUP_PATH = Path("Data/benchmarks/medical_benchmark_with_ground_truth.backup.json")

SYSTEM_PROMPT = """You are a board-certified medical editor creating gold-standard reference answers for a medical benchmark based on The Gale Encyclopedia of Medicine.

Requirements for Ground Truth:
1. Provide a concise, highly accurate, 1-3 sentence factual answer.
2. Directly answer the clinical question with specific medical terminology (etiologies, clinical signs, diagnostics, or first-line therapeutics).
3. Do NOT include conversational filler, disclaimers, or introductory phrases like "According to The Gale Encyclopedia of Medicine" or "As an AI".
4. Be authoritative, objective, and directly answerable from a medical reference encyclopedia.
"""

FEW_SHOT_EXAMPLES = [
    {
        "query": "What are the primary symptoms and clinical signs of asthma?",
        "ground_truth": "Primary symptoms of asthma include recurrent wheezing, shortness of breath (dyspnea), persistent coughing (especially at night), and chest tightness caused by chronic airway inflammation and reversible bronchial constriction.",
    },
    {
        "query": "What environmental triggers can precipitate an acute asthma attack?",
        "ground_truth": "Environmental triggers for acute asthma include airborne allergens (pollen, dust mites, pet dander), tobacco smoke, cold dry air, respiratory infections, exercise, and chemical irritants.",
    },
    {
        "query": "What bronchodilator and corticosteroid medications are used to manage asthma?",
        "ground_truth": "Asthma management utilizes bronchodilators such as short-acting beta-agonists (Albuterol) for immediate relief, and daily inhaled corticosteroids (fluticasone, budesonide) as anti-inflammatory controller medications.",
    },
]


def is_template(text: str) -> bool:
    indicators = [
        "involves specific physiological etiologies, characteristic clinical signs, diagnostic evaluation, and evidence-based therapeutic management protocols",
        "According to The Gale Encyclopedia of Medicine,",
    ]
    return any(ind in text for ind in indicators)


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment or .env!")
        sys.exit(1)

    if not BENCHMARK_PATH.exists():
        logger.error("Benchmark file not found: %s", BENCHMARK_PATH)
        sys.exit(1)

    # 1. Backup original file if backup doesn't exist
    if not BACKUP_PATH.exists():
        shutil.copyfile(BENCHMARK_PATH, BACKUP_PATH)
        logger.info("Created backup at: %s", BACKUP_PATH)
    else:
        logger.info("Backup already exists at: %s", BACKUP_PATH)

    with BENCHMARK_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions", [])
    logger.info("Total questions in benchmark: %d", len(questions))

    client = OpenAI(api_key=api_key)

    updated_count = 0
    for idx, q in enumerate(questions, 1):
        gt = q.get("ground_truth", "")
        qid = q.get("id", f"Q{idx}")
        category = q.get("category", "")
        query = q.get("query", "")

        if is_template(gt):
            logger.info("[%d/%d] Generating GT for [%s] (%s): %s", idx, len(questions), qid, category, query)

            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for ex in FEW_SHOT_EXAMPLES:
                messages.append({"role": "user", "content": f"Query: {ex['query']}"})
                messages.append({"role": "assistant", "content": ex["ground_truth"]})
            messages.append({"role": "user", "content": f"Medical Specialty: {category}\nQuery: {query}"})

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.0,
                    max_tokens=250,
                )
                new_gt = response.choices[0].message.content.strip()
                q["ground_truth"] = new_gt
                updated_count += 1
                logger.info("  -> Generated: %s\n", new_gt)
            except Exception as e:
                logger.error("Failed to generate GT for %s: %s", qid, e)

    # Save updated file
    with BENCHMARK_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info("Successfully updated %d questions in %s", updated_count, BENCHMARK_PATH)


if __name__ == "__main__":
    main()
