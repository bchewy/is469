from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path


ASCII_WORD_RE = re.compile(r"[A-Za-z]{4,}")
META_PATTERNS = [
    re.compile(r"This translation", re.I),
    re.compile(r"natural .*Japanese", re.I),
    re.compile(r"more accurate", re.I),
    re.compile(r"would be more natural", re.I),
    re.compile(r"translates to", re.I),
]


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_contaminated(text: str) -> bool:
    if ASCII_WORD_RE.search(text):
        return True
    if any(pattern.search(text) for pattern in META_PATTERNS):
        return True
    if "\n" in text:
        return True
    if len(text) > 220:
        return True
    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare baseline and fine-tuned EN->JA output JSONL files."
    )
    parser.add_argument("--reference", required=True, help="Reference eval JSONL path")
    parser.add_argument("--baseline", required=True, help="Baseline outputs JSONL path")
    parser.add_argument("--candidate", required=True, help="Fine-tuned outputs JSONL path")
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="How many best/worst changed examples to print",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    ref_rows = load_jsonl(Path(args.reference))
    baseline_rows = load_jsonl(Path(args.baseline))
    candidate_rows = load_jsonl(Path(args.candidate))

    refs = {row["id"]: row for row in ref_rows}
    if len(baseline_rows) != len(candidate_rows):
        raise ValueError(
            f"Row count mismatch: baseline={len(baseline_rows)} candidate={len(candidate_rows)}"
        )

    summary = {
        "num_samples": len(baseline_rows),
        "same_output_count": 0,
        "s1_better_count": 0,
        "s0_better_count": 0,
        "same_quality_count": 0,
        "s1_exact_gain": 0,
        "s0_exact_gain": 0,
        "avg_char_similarity_s0": 0.0,
        "avg_char_similarity_s1": 0.0,
        "avg_latency_s0_ms": 0.0,
        "avg_latency_s1_ms": 0.0,
        "s0_contaminated_count": 0,
        "s1_contaminated_count": 0,
        "s0_cleaned_by_s1": 0,
        "s1_cleaned_by_s0": 0,
    }
    changed_examples: list[dict] = []

    for baseline, candidate in zip(baseline_rows, candidate_rows):
        if baseline["id"] != candidate["id"]:
            raise ValueError(
                f"ID mismatch: baseline={baseline['id']} candidate={candidate['id']}"
            )

        ref = refs[baseline["id"]]["target_ja"]
        s0_text = baseline["prediction_ja"]
        s1_text = candidate["prediction_ja"]

        if s0_text == s1_text:
            summary["same_output_count"] += 1

        s0_score = SequenceMatcher(None, s0_text, ref).ratio()
        s1_score = SequenceMatcher(None, s1_text, ref).ratio()
        summary["avg_char_similarity_s0"] += s0_score
        summary["avg_char_similarity_s1"] += s1_score
        summary["avg_latency_s0_ms"] += float(baseline.get("latency_ms", 0) or 0)
        summary["avg_latency_s1_ms"] += float(candidate.get("latency_ms", 0) or 0)

        if abs(s1_score - s0_score) < 1e-12:
            summary["same_quality_count"] += 1
        elif s1_score > s0_score:
            summary["s1_better_count"] += 1
        else:
            summary["s0_better_count"] += 1

        if s1_text == ref and s0_text != ref:
            summary["s1_exact_gain"] += 1
        if s0_text == ref and s1_text != ref:
            summary["s0_exact_gain"] += 1

        s0_bad = is_contaminated(s0_text)
        s1_bad = is_contaminated(s1_text)
        if s0_bad:
            summary["s0_contaminated_count"] += 1
        if s1_bad:
            summary["s1_contaminated_count"] += 1
        if s0_bad and not s1_bad:
            summary["s0_cleaned_by_s1"] += 1
        if s1_bad and not s0_bad:
            summary["s1_cleaned_by_s0"] += 1

        if s0_text != s1_text:
            changed_examples.append(
                {
                    "id": baseline["id"],
                    "source_en": baseline["source_en"],
                    "reference_ja": ref,
                    "s0": s0_text,
                    "s1": s1_text,
                    "s0_ratio": round(s0_score, 4),
                    "s1_ratio": round(s1_score, 4),
                    "delta": round(s1_score - s0_score, 4),
                }
            )

    n = max(summary["num_samples"], 1)
    summary["same_output_pct"] = round(summary["same_output_count"] / n, 4)
    summary["avg_char_similarity_s0"] = round(summary["avg_char_similarity_s0"] / n, 4)
    summary["avg_char_similarity_s1"] = round(summary["avg_char_similarity_s1"] / n, 4)
    summary["avg_latency_s0_ms"] = round(summary["avg_latency_s0_ms"] / n, 1)
    summary["avg_latency_s1_ms"] = round(summary["avg_latency_s1_ms"] / n, 1)

    changed_examples.sort(key=lambda row: row["delta"])
    worst = changed_examples[: args.top_k]
    best = list(reversed(changed_examples[-args.top_k :]))

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nBEST_DELTA")
    for row in best:
        print(json.dumps(row, ensure_ascii=False))
    print("\nWORST_DELTA")
    for row in worst:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
