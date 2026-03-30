#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.schemas import TranslationRow, write_rows


def _normalize_en(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _is_reasonable_pair(source_en: str, target_ja: str) -> bool:
    if not source_en or not target_ja:
        return False
    if len(source_en) < 2 or len(target_ja) < 1:
        return False
    if len(source_en) > 280 or len(target_ja) > 280:
        return False
    return True


def load_unique_rows(tsv_path: Path) -> list[TranslationRow]:
    rows: list[TranslationRow] = []
    seen_en: set[str] = set()

    with tsv_path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue

            row_id = parts[0].strip() or f"engjap-{line_no}"
            source_en = parts[1].strip()
            target_ja = parts[3].strip()

            if not _is_reasonable_pair(source_en, target_ja):
                continue

            en_key = _normalize_en(source_en)
            if en_key in seen_en:
                continue

            seen_en.add(en_key)
            rows.append(
                TranslationRow(
                    id=f"engjap-{row_id}",
                    source_en=source_en,
                    target_ja=target_ja,
                    domain="general",
                    source_ref="eng-jap.tsv",
                    quality_score=0.9,
                    license="unknown",
                    group_key=en_key,
                )
            )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build small train/dev/test splits from kb/eng-jap.tsv"
    )
    parser.add_argument("--input", default="kb/eng-jap.tsv")
    parser.add_argument("--output-dir", default="data/splits")
    parser.add_argument("--train-count", type=int, default=250)
    parser.add_argument("--dev-count", type=int, default=250)
    parser.add_argument("--test-count", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tsv_path = Path(args.input)
    if not tsv_path.exists():
        raise FileNotFoundError(f"Input TSV not found: {tsv_path}")

    rows = load_unique_rows(tsv_path)
    total_needed = args.train_count + args.dev_count + args.test_count
    if len(rows) < total_needed:
        raise ValueError(
            f"Not enough unique rows in {tsv_path}: need {total_needed}, found {len(rows)}"
        )

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    chosen = rows[:total_needed]

    train_rows = chosen[: args.train_count]
    dev_rows = chosen[args.train_count : args.train_count + args.dev_count]
    test_rows = chosen[args.train_count + args.dev_count :]

    for split_name, split_rows in (
        ("train", train_rows),
        ("dev", dev_rows),
        ("test", test_rows),
    ):
        for row in split_rows:
            row.split = split_name

    out_dir = Path(args.output_dir)
    write_rows(train_rows, out_dir / "train_v1.jsonl")
    write_rows(dev_rows, out_dir / "dev_v1.jsonl")
    write_rows(test_rows, out_dir / "test_v1.jsonl")

    print(
        f"Wrote {len(train_rows)} train, {len(dev_rows)} dev, {len(test_rows)} test rows to {out_dir}"
    )


if __name__ == "__main__":
    main()
