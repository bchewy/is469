#!/usr/bin/env python3
"""Curate a translation memory from the best training pairs.

Selects high-quality sentence pairs based on length, completeness,
and diversity. Adds topic tags via simple keyword heuristics.

Usage:
    python -m scripts.build_translation_memory \
        --input data/splits/train_v1.jsonl \
        --output kb/translation_memory.jsonl \
        --max-entries 500
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TOPIC_KEYWORDS = {
    "business": ["company", "business", "market", "industry", "corporate", "profit", "revenue", "management"],
    "technology": ["software", "computer", "digital", "internet", "system", "data", "network", "server", "technology"],
    "politics": ["government", "policy", "political", "election", "minister", "parliament", "law", "regulation"],
    "economy": ["economy", "economic", "financial", "bank", "investment", "trade", "gdp", "inflation"],
    "society": ["society", "social", "community", "culture", "population", "education", "public"],
    "environment": ["environment", "climate", "energy", "pollution", "renewable", "carbon", "nature"],
    "health": ["health", "medical", "hospital", "disease", "treatment", "patient", "doctor"],
    "science": ["research", "study", "scientific", "experiment", "discovery", "university"],
    "legal": ["court", "judge", "legal", "law", "rights", "criminal", "justice", "contract"],
    "daily_life": ["food", "home", "family", "school", "travel", "shopping", "restaurant"],
}


def _detect_topic(en_text: str) -> str:
    lower = en_text.lower()
    scores = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        scores[topic] = sum(1 for kw in keywords if kw in lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def _quality_score(en: str, ja: str) -> float:
    """Higher = better quality pair for TM."""
    score = 0.0
    if 20 <= len(en) <= 300:
        score += 1.0
    elif len(en) > 300:
        score += 0.5
    if en.rstrip().endswith((".", "!", "?", ":")):
        score += 0.5
    ratio = len(en) / max(len(ja), 1)
    if 1.5 <= ratio <= 4.0:
        score += 1.0
    if re.search(r"[ぁ-ん]", ja) and re.search(r"[ァ-ヶ]|[一-龥]", ja):
        score += 0.5
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description="Build translation memory")
    parser.add_argument("--input", default="data/splits/train_v1.jsonl")
    parser.add_argument("--output", default="kb/translation_memory.jsonl")
    parser.add_argument("--max-entries", type=int, default=500)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}")
        sys.exit(1)

    rows = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"Loaded {len(rows)} rows from {input_path}")

    scored = []
    for row in rows:
        en = row.get("source_en", "")
        ja = row.get("target_ja", "")
        if len(en) < 15 or len(ja) < 5:
            continue
        qs = _quality_score(en, ja)
        scored.append((qs, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = scored[: args.max_entries]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    topic_counts: dict[str, int] = {}
    with out_path.open("w", encoding="utf-8") as f:
        for _, row in selected:
            en = row["source_en"]
            ja = row["target_ja"]
            topic = _detect_topic(en)
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
            entry = {
                "source_en": en,
                "target_ja": ja,
                "topic": topic,
                "source_ref": row.get("source_ref", ""),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Wrote {len(selected)} TM entries to {out_path}")
    print("Topic distribution:")
    for topic, count in sorted(topic_counts.items(), key=lambda x: -x[1]):
        print(f"  {topic}: {count}")


if __name__ == "__main__":
    main()
