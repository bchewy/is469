#!/usr/bin/env python3
"""Normalize and filter raw EN-JA pairs into a clean processed JSONL.

Pipeline:
  1. Unicode normalization (NFKC)
  2. Whitespace / punctuation cleanup
  3. Language-ID heuristic checks (has_kana for JA, is_ascii_dominant for EN)
  4. Length-ratio filtering
  5. Exact and near-duplicate removal (hash-based)
  6. Quality-score floor

Usage:
    python -m scripts.normalize_and_filter_pairs \
        --input data/raw/combined_raw.jsonl \
        --output data/processed/pilot_v1.jsonl \
        --min-en-chars 4 --min-ja-chars 2 --max-len-ratio 9.0 --min-quality 0.5
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.schemas import TranslationRow, load_rows, write_rows


# ── Text normalization ───────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()
    return text


# ── Language-ID heuristics ───────────────────────────────────────────────────

_KANA_RE = re.compile(r"[\u3040-\u309F\u30A0-\u30FF]")
_CJK_RE = re.compile(r"[\u4E00-\u9FFF]")


def looks_japanese(text: str) -> bool:
    """True if text contains kana or CJK characters."""
    return bool(_KANA_RE.search(text)) or bool(_CJK_RE.search(text))


def looks_english(text: str) -> bool:
    """True if >50% of alphanumeric chars are ASCII."""
    alnum = [c for c in text if c.isalnum()]
    if not alnum:
        return False
    ascii_count = sum(1 for c in alnum if ord(c) < 128)
    return ascii_count / len(alnum) > 0.5


# ── Dedup ────────────────────────────────────────────────────────────────────

def _pair_hash(en: str, ja: str) -> str:
    combined = f"{en.lower().strip()}|||{ja.strip()}"
    return hashlib.md5(combined.encode("utf-8")).hexdigest()


def _ngram_hash(text: str, n: int = 3) -> str:
    """Coarse near-dup fingerprint via character trigrams."""
    text = re.sub(r"\s+", "", text.lower())
    trigrams = sorted(set(text[i : i + n] for i in range(len(text) - n + 1)))
    return hashlib.md5("".join(trigrams).encode("utf-8")).hexdigest()


# ── Main pipeline ────────────────────────────────────────────────────────────

def filter_and_normalize(
    rows: list[TranslationRow],
    *,
    min_en_chars: int = 4,
    min_ja_chars: int = 2,
    max_len_ratio: float = 9.0,
    min_quality: float = 0.5,
) -> tuple[list[TranslationRow], dict[str, int]]:

    stats = {
        "input": len(rows),
        "empty_text": 0,
        "lang_id_fail": 0,
        "too_short": 0,
        "len_ratio_fail": 0,
        "quality_floor": 0,
        "exact_dup": 0,
        "near_dup": 0,
        "output": 0,
    }

    seen_exact: set[str] = set()
    seen_near: set[str] = set()
    kept: list[TranslationRow] = []

    for row in rows:
        en = normalize_text(row.source_en)
        ja = normalize_text(row.target_ja)

        if not en or not ja:
            stats["empty_text"] += 1
            continue

        if not looks_english(en) or not looks_japanese(ja):
            stats["lang_id_fail"] += 1
            continue

        if len(en) < min_en_chars or len(ja) < min_ja_chars:
            stats["too_short"] += 1
            continue

        ratio = max(len(en), len(ja)) / max(len(ja), len(en), 1)
        if ratio > max_len_ratio:
            stats["len_ratio_fail"] += 1
            continue

        if row.quality_score < min_quality:
            stats["quality_floor"] += 1
            continue

        ph = _pair_hash(en, ja)
        if ph in seen_exact:
            stats["exact_dup"] += 1
            continue
        seen_exact.add(ph)

        nh = _ngram_hash(en + ja)
        if nh in seen_near:
            stats["near_dup"] += 1
            continue
        seen_near.add(nh)

        row.source_en = en
        row.target_ja = ja
        kept.append(row)

    stats["output"] = len(kept)
    return kept, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize & filter EN-JA pairs")
    parser.add_argument("--input", required=True, help="Path to raw JSONL")
    parser.add_argument(
        "--output", default="data/processed/pilot_v1.jsonl", help="Output path"
    )
    parser.add_argument("--min-en-chars", type=int, default=4)
    parser.add_argument("--min-ja-chars", type=int, default=2)
    parser.add_argument("--max-len-ratio", type=float, default=9.0)
    parser.add_argument("--min-quality", type=float, default=0.5)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    rows = load_rows(input_path)
    print(f"Loaded {len(rows)} raw rows from {input_path}")

    kept, stats = filter_and_normalize(
        rows,
        min_en_chars=args.min_en_chars,
        min_ja_chars=args.min_ja_chars,
        max_len_ratio=args.max_len_ratio,
        min_quality=args.min_quality,
    )

    n = write_rows(kept, args.output)
    print(f"\nWrote {n} clean rows to {args.output}")
    print("Filter stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
