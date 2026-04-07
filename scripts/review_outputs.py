import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a qualitative review from s3 outputs JSONL")
    p.add_argument(
        "--input",
        default="results/s3_outputs_smoke50.jsonl",
        help="Path to s3 outputs JSONL (default: results/s3_outputs_smoke50.jsonl)",
    )
    return p.parse_args()


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.input))

    categories: dict[str, list[dict]] = {
        "terminology_wins": [],
        "retrieval_failures": [],
        "correctly_caught_errors": [],
        "major_mistranslations": [],
        "agent_rewrites_helped": [],
    }

    for r in rows:
        term = r.get("terminology_eval") or {}
        ret = r.get("retrieval_eval") or {}
        err = r.get("error_check") or {}
        gold = r.get("gold_error_label") or {}
        trace = r.get("agent_trace") or []

        if term.get("term_count", 0) > 0 and term.get("accuracy", 0) == 1.0:
            categories["terminology_wins"].append(r)

        if ret.get("expected_target_count", 0) > 0 and not ret.get("hit_at_k"):
            categories["retrieval_failures"].append(r)

        if gold.get("has_error") and err.get("has_error"):
            categories["correctly_caught_errors"].append(r)

        if gold.get("severity") == "major":
            categories["major_mistranslations"].append(r)

        scores = [s.get("critic_coverage_score") for s in trace if "critic_coverage_score" in s]
        scores = [s for s in scores if s is not None]
        if len(scores) >= 2 and scores[-1] > scores[0]:
            categories["agent_rewrites_helped"].append(r)

    for cat, items in categories.items():
        print(f"\n## {cat} ({len(items)} rows)")
        for row in items[:3]:
            print(f"- ID: {row.get('id')}")
            print(f"  EN: {row.get('source_en')}")
            print(f"  JA: {row.get('prediction_ja')}")
            print(f"  REF: {row.get('reference_ja')}")
            if row.get("gold_error_label"):
                print(f"  GOLD: {row['gold_error_label']}")
            print("  Commentary: ")


if __name__ == "__main__":
    main()
