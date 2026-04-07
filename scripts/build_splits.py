#!/usr/bin/env python3
"""Build deterministic train/dev/test splits with anti-leakage grouping.

Groups rows by source_ref (or group_key if set) so that all rows from the same
document / source stay in the same split.  Falls back to row-level splitting
when group_key and source_ref are both empty.

Usage:
    python -m scripts.build_splits \
        --input data/processed/pilot_v1.jsonl \
        --output-dir data/splits \
        --train-ratio 0.8 --dev-ratio 0.1 --test-ratio 0.1 \
        --seed 42
"""
from __future__ import annotations

import argparse
import random
import sys
import math
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.schemas import TranslationRow, load_rows, write_rows, validate_jsonl_file


def _group_key(row: TranslationRow) -> str:
    if row.group_key:
        return row.group_key
    if row.source_ref:
        return row.source_ref
    return row.id


def _parse_source_train_quotas(spec: str | None) -> dict[str, int]:
    if not spec:
        return {}

    quotas: dict[str, int] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid source quota '{item}', expected source=count")
        source, raw_count = item.split("=", 1)
        quotas[source.strip()] = int(raw_count.strip())
    return quotas


def build_splits(
    rows: list[TranslationRow],
    *,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    source_train_quotas: dict[str, int] | None = None,
    allow_empty_test: bool = False,
) -> dict[str, list[TranslationRow]]:
    total = train_ratio + dev_ratio + test_ratio
    train_ratio /= total
    dev_ratio /= total
    test_ratio /= total

    if source_train_quotas:
        rng = random.Random(seed)
        by_source: dict[str, list[TranslationRow]] = defaultdict(list)
        for row in rows:
            by_source[row.source_ref].append(row)

        unexpected_sources = sorted(
            source_ref
            for source_ref in by_source
            if source_ref not in source_train_quotas
        )
        if unexpected_sources:
            raise ValueError(
                "source_train_quotas does not cover all input sources: "
                + ", ".join(unexpected_sources)
            )

        splits: dict[str, list[TranslationRow]] = {"train": [], "dev": [], "test": []}
        for source_ref, train_quota in source_train_quotas.items():
            source_rows = list(by_source.get(source_ref, []))
            rng.shuffle(source_rows)

            train_take = min(train_quota, len(source_rows))
            remaining = len(source_rows) - train_take

            dev_target = 0
            test_target = 0
            if train_ratio > 0:
                if dev_ratio > 0:
                    dev_target = math.ceil(train_quota * dev_ratio / train_ratio)
                    if dev_target == 0 and train_take > 0 and remaining > 0:
                        dev_target = 1
                if test_ratio > 0:
                    test_target = math.ceil(train_quota * test_ratio / train_ratio)
                    if test_target == 0 and train_take > 0 and remaining > dev_target:
                        test_target = 1

            dev_take = min(dev_target, remaining)
            remaining -= dev_take
            test_take = min(test_target, remaining)

            train_rows = source_rows[:train_take]
            dev_rows = source_rows[train_take : train_take + dev_take]
            test_rows = source_rows[train_take + dev_take : train_take + dev_take + test_take]

            for row in train_rows:
                row.split = "train"
            for row in dev_rows:
                row.split = "dev"
            for row in test_rows:
                row.split = "test"

            splits["train"].extend(train_rows)
            splits["dev"].extend(dev_rows)
            splits["test"].extend(test_rows)

        for split_name in splits:
            rng.shuffle(splits[split_name])

        return splits

    groups: dict[str, list[TranslationRow]] = defaultdict(list)
    for row in rows:
        groups[_group_key(row)].append(row)

    n = len(rows)
    max_group_size = max(len(v) for v in groups.values()) if groups else 0

    if max_group_size > n * 0.5 and len(groups) < 10:
        print("  Groups too coarse for source-level anti-leakage, falling back to row-level splitting")
        groups = {row.id: [row] for row in rows}

    group_keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    splits: dict[str, list[TranslationRow]] = {"train": [], "dev": [], "test": []}
    cumulative = 0

    for gk in group_keys:
        group_rows = groups[gk]
        frac = cumulative / n if n else 0

        if frac < train_ratio:
            split = "train"
        elif frac < train_ratio + dev_ratio:
            split = "dev"
        else:
            split = "test"

        for row in group_rows:
            row.split = split
        splits[split].extend(group_rows)
        cumulative += len(group_rows)

    if not splits["dev"] and splits["train"]:
        moved = splits["train"][-1:]
        splits["train"] = splits["train"][:-1]
        for r in moved:
            r.split = "dev"
        splits["dev"] = moved

    if not splits["test"] and splits["train"] and not allow_empty_test:
        moved = splits["train"][-1:]
        splits["train"] = splits["train"][:-1]
        for r in moved:
            r.split = "test"
        splits["test"] = moved

    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/dev/test splits")
    parser.add_argument("--input", required=True, help="Processed JSONL")
    parser.add_argument("--output-dir", default="data/splits")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", default="v1", help="Split version suffix")
    parser.add_argument(
        "--source-train-quotas",
        default="",
        help="Optional comma-separated train quotas, e.g. jparacrawl=18000,hf_tatoeba=4000",
    )
    parser.add_argument(
        "--allow-empty-test",
        action="store_true",
        help="Allow zero-row test output when test_ratio is 0",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}")
        sys.exit(1)

    rows = load_rows(input_path)
    print(f"Loaded {len(rows)} rows from {input_path}")

    splits = build_splits(
        rows,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        source_train_quotas=_parse_source_train_quotas(args.source_train_quotas),
        allow_empty_test=args.allow_empty_test,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_rows in splits.items():
        out_path = out_dir / f"{split_name}_{args.version}.jsonl"
        n = write_rows(split_rows, out_path)
        print(f"  {split_name}: {n} rows -> {out_path}")

    print("\nAnti-leakage check:")
    train_ids = {r.id for r in splits["train"]}
    dev_ids = {r.id for r in splits["dev"]}
    test_ids = {r.id for r in splits["test"]}
    td = train_ids & dev_ids
    tt = train_ids & test_ids
    dt = dev_ids & test_ids
    if td or tt or dt:
        print(f"  WARNING: ID overlap – train∩dev={len(td)}, train∩test={len(tt)}, dev∩test={len(dt)}")
    else:
        print("  No ID overlap across splits.")

    for split_name in ("train", "dev", "test"):
        out_path = out_dir / f"{split_name}_{args.version}.jsonl"
        valid, errors = validate_jsonl_file(out_path, require_split=True, expected_split=split_name)
        if errors:
            print(f"  Validation errors in {out_path}: {errors[:5]}")
        else:
            print(f"  {out_path}: {valid} rows valid")


if __name__ == "__main__":
    main()
