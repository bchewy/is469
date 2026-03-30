#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


def load_eng_jap_rows(tsv_path: Path) -> list[TranslationRow]:
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


def load_annotation_rows(jsonl_path: Path) -> list[TranslationRow]:
    rows: list[TranslationRow] = []
    seen_en: set[str] = set()

    with jsonl_path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            source_en = str(obj.get("source_en", "")).strip()
            target_ja = str(obj.get("reference_ja", "")).strip()
            row_id = str(obj.get("id", "")).strip()

            if not _is_reasonable_pair(source_en, target_ja):
                continue

            en_key = _normalize_en(source_en)
            if en_key in seen_en:
                continue

            seen_en.add(en_key)
            rows.append(
                TranslationRow(
                    id=f"annot-{row_id}" if row_id else f"annot-{len(rows)}",
                    source_en=source_en,
                    target_ja=target_ja,
                    domain="general",
                    source_ref=jsonl_path.name,
                    quality_score=0.95,
                    license="unknown",
                    group_key=en_key,
                )
            )

    return rows


def load_translation_memory_rows(jsonl_path: Path) -> list[TranslationRow]:
    rows: list[TranslationRow] = []

    with jsonl_path.open("r", encoding="utf-8-sig") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            source_en = str(obj.get("source_en", "")).strip()
            target_ja = str(obj.get("target_ja", "")).strip()
            topic = str(obj.get("topic", "")).strip()

            if not _is_reasonable_pair(source_en, target_ja):
                continue

            rows.append(
                TranslationRow(
                    id=f"tm-{idx}",
                    source_en=source_en,
                    target_ja=target_ja,
                    domain="ui_strings" if topic else "general",
                    source_ref="translation_memory.jsonl",
                    quality_score=1.0,
                    license="unknown",
                    group_key=_normalize_en(source_en),
                )
            )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a more realistic mixed train/dev/test split from kb/"
    )
    parser.add_argument("--eng-jap", default="kb/eng-jap.tsv")
    parser.add_argument("--annotations", default="kb/annotations_raw.jsonl")
    parser.add_argument("--translation-memory", default="kb/translation_memory.jsonl")
    parser.add_argument("--output-dir", default="data/splits")
    parser.add_argument("--train-count", type=int, default=2000)
    parser.add_argument("--dev-count", type=int, default=250)
    parser.add_argument("--test-count", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    eng_rows = load_eng_jap_rows(Path(args.eng_jap))
    annotation_rows = load_annotation_rows(Path(args.annotations))
    tm_rows = load_translation_memory_rows(Path(args.translation_memory))

    rng.shuffle(annotation_rows)
    if len(annotation_rows) < args.dev_count + args.test_count:
        raise ValueError(
            f"Need at least {args.dev_count + args.test_count} annotation rows, "
            f"found {len(annotation_rows)}"
        )

    dev_rows = annotation_rows[: args.dev_count]
    test_rows = annotation_rows[args.dev_count : args.dev_count + args.test_count]

    held_out_en = {_normalize_en(r.source_en) for r in dev_rows + test_rows}
    tm_en = {_normalize_en(r.source_en) for r in tm_rows}

    train_pool = [
        r
        for r in eng_rows
        if _normalize_en(r.source_en) not in held_out_en
        and _normalize_en(r.source_en) not in tm_en
    ]
    rng.shuffle(train_pool)

    train_rows: list[TranslationRow] = []
    train_rows.extend(tm_rows)

    remaining_needed = max(0, args.train_count - len(train_rows))
    if len(train_pool) < remaining_needed:
        raise ValueError(
            f"Need {remaining_needed} eng-jap rows after filtering, found {len(train_pool)}"
        )
    train_rows.extend(train_pool[:remaining_needed])

    rng.shuffle(train_rows)

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
        "Wrote "
        f"{len(train_rows)} train rows "
        f"({sum(1 for r in train_rows if r.source_ref == 'eng-jap.tsv')} from eng-jap.tsv, "
        f"{sum(1 for r in train_rows if r.source_ref == 'translation_memory.jsonl')} from translation_memory.jsonl), "
        f"{len(dev_rows)} dev rows, and {len(test_rows)} test rows from annotations_raw.jsonl "
        f"to {out_dir}"
    )


if __name__ == "__main__":
    main()
