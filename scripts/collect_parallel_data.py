#!/usr/bin/env python3
"""Collect EN-JA parallel pairs from public sources and local imports.

Outputs raw JSONL under data/raw/ with provenance metadata.

Usage:
    python -m scripts.collect_parallel_data --sources tatoeba,jesc,local --local-file extra.csv
    python -m scripts.collect_parallel_data --sources tatoeba --max-per-source 5000
    python -m scripts.collect_parallel_data --validate-only data/raw/tatoeba_raw.jsonl
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.schemas import TranslationRow, validate_jsonl_file, write_rows

RAW_DIR = Path("data/raw")

# ── Source: Tatoeba ──────────────────────────────────────────────────────────

TATOEBA_URL = (
    "https://downloads.tatoeba.org/exports/per_language/jpn/jpn-eng/sentences.tsv"
)


def _download_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": "genai-llms/0.1"})
    with urlopen(req, timeout=120) as resp:
        return resp.read().decode("utf-8")


def collect_tatoeba(max_rows: int) -> list[TranslationRow]:
    """Pull from Tatoeba jpn-eng sentence pairs export (TSV)."""
    print(f"[tatoeba] Downloading from {TATOEBA_URL} ...")
    try:
        tsv = _download_text(TATOEBA_URL)
    except Exception as exc:
        print(f"[tatoeba] Could not download Tatoeba TSV – falling back to API: {exc}")
        return _collect_tatoeba_api(max_rows)

    rows: list[TranslationRow] = []
    for line in tsv.splitlines():
        if len(rows) >= max_rows:
            break
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        ja_text = parts[1].strip()
        en_text = parts[3].strip()
        if not ja_text or not en_text:
            continue
        rows.append(
            TranslationRow(
                id=TranslationRow.generate_id("tat"),
                source_en=en_text,
                target_ja=ja_text,
                domain="general",
                source_ref="tatoeba",
                quality_score=0.8,
                license="cc-by-4.0",
            )
        )
    print(f"[tatoeba] Collected {len(rows)} pairs")
    return rows


def _collect_tatoeba_api(max_rows: int) -> list[TranslationRow]:
    """Lightweight fallback using Tatoeba search API."""
    rows: list[TranslationRow] = []
    page = 1
    per_page = min(max_rows, 100)
    while len(rows) < max_rows and page <= 20:
        url = (
            f"https://tatoeba.org/eng/api_v0/search"
            f"?from=eng&to=jpn&page={page}&limit={per_page}"
        )
        try:
            data = json.loads(_download_text(url))
        except Exception:
            break
        results = data.get("results", [])
        if not results:
            break
        for r in results:
            if len(rows) >= max_rows:
                break
            en_text = r.get("text", "").strip()
            translations = r.get("translations", [[]])
            ja_text = ""
            for group in translations:
                for t in group:
                    if t.get("lang") == "jpn":
                        ja_text = t.get("text", "").strip()
                        break
                if ja_text:
                    break
            if en_text and ja_text:
                rows.append(
                    TranslationRow(
                        id=TranslationRow.generate_id("tat"),
                        source_en=en_text,
                        target_ja=ja_text,
                        domain="general",
                        source_ref="tatoeba",
                        quality_score=0.8,
                        license="cc-by-4.0",
                    )
                )
        page += 1
    print(f"[tatoeba-api] Collected {len(rows)} pairs")
    return rows


# ── Source: HuggingFace Tatoeba ───────────────────────────────────────────────


def collect_hf_tatoeba(max_rows: int) -> list[TranslationRow]:
    """Pull EN-JA pairs from the HuggingFace Tatoeba dataset mirror."""
    print("[hf_tatoeba] Loading Helsinki-NLP/tatoeba_mt eng-jpn from HuggingFace ...")
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "Helsinki-NLP/tatoeba_mt",
            "eng-jpn",
            split="test",
        )
    except Exception as exc:
        print(f"[hf_tatoeba] Failed to load dataset: {exc}")
        return []

    rows: list[TranslationRow] = []
    for i, item in enumerate(ds):
        if len(rows) >= max_rows:
            break
        translation = item.get("translation", {})
        en_text = translation.get("eng", "").strip()
        ja_text = translation.get("jpn", "").strip()
        if en_text and ja_text:
            rows.append(
                TranslationRow(
                    id=TranslationRow.generate_id("hftat"),
                    source_en=en_text,
                    target_ja=ja_text,
                    domain="general",
                    source_ref="hf_tatoeba",
                    quality_score=0.8,
                    license="cc-by-4.0",
                )
            )
    print(f"[hf_tatoeba] Collected {len(rows)} pairs")
    return rows


# ── Source: OPUS-100 via HuggingFace ─────────────────────────────────────────


def collect_opus100(max_rows: int) -> list[TranslationRow]:
    """Pull EN-JA pairs from OPUS-100 (large multilingual parallel corpus)."""
    print("[opus100] Loading Helsinki-NLP/opus-100 en-ja from HuggingFace ...")
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "Helsinki-NLP/opus-100",
            "en-ja",
            split="train",
        )
    except Exception as exc:
        print(f"[opus100] Failed to load dataset: {exc}")
        return []

    rows: list[TranslationRow] = []
    for item in ds:
        if len(rows) >= max_rows:
            break
        translation = item.get("translation", {})
        en_text = translation.get("en", "").strip()
        ja_text = translation.get("ja", "").strip()
        if en_text and ja_text:
            rows.append(
                TranslationRow(
                    id=TranslationRow.generate_id("opus"),
                    source_en=en_text,
                    target_ja=ja_text,
                    domain="general",
                    source_ref="opus100",
                    quality_score=0.75,
                    license="cc-by-4.0",
                )
            )
    print(f"[opus100] Collected {len(rows)} pairs")
    return rows


# ── Source: JESC via HuggingFace ─────────────────────────────────────────────


def collect_jesc(max_rows: int) -> list[TranslationRow]:
    """Pull from JESC (Japanese-English Subtitle Corpus) via HuggingFace."""
    print("[jesc] Loading jesc dataset from HuggingFace ...")
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "Hoshikuzu/jesc",
            split="train",
        )
    except Exception as exc:
        print(f"[jesc] HF load failed, trying direct download: {exc}")
        return _collect_jesc_direct(max_rows)

    rows: list[TranslationRow] = []
    for item in ds:
        if len(rows) >= max_rows:
            break
        en_text = item.get("en", item.get("english", "")).strip()
        ja_text = item.get("ja", item.get("japanese", "")).strip()
        if en_text and ja_text:
            rows.append(
                TranslationRow(
                    id=TranslationRow.generate_id("jesc"),
                    source_en=en_text,
                    target_ja=ja_text,
                    domain="general",
                    source_ref="jesc",
                    quality_score=0.7,
                    license="cc-by-4.0",
                )
            )
    print(f"[jesc] Collected {len(rows)} pairs")
    return rows


def _collect_jesc_direct(max_rows: int) -> list[TranslationRow]:
    """Fallback: download JESC tar.gz directly."""
    import tarfile
    jesc_url = "https://nlp.stanford.edu/projects/jesc/data/split.tar.gz"
    print(f"[jesc-direct] Downloading from {jesc_url} ...")
    try:
        req = Request(jesc_url, headers={"User-Agent": "genai-llms/0.1"})
        with urlopen(req, timeout=180) as resp:
            raw = resp.read()
    except Exception as exc:
        print(f"[jesc-direct] Download failed: {exc}")
        return []

    rows: list[TranslationRow] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
            for member in tar.getmembers():
                if not member.name.endswith(".tsv") and not member.name.endswith(".txt"):
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                for bline in f:
                    if len(rows) >= max_rows:
                        break
                    line = bline.decode("utf-8", errors="replace").strip()
                    parts = line.split("\t")
                    if len(parts) < 2:
                        continue
                    en_text, ja_text = parts[0].strip(), parts[1].strip()
                    if not en_text or not ja_text:
                        continue
                    rows.append(
                        TranslationRow(
                            id=TranslationRow.generate_id("jesc"),
                            source_en=en_text,
                            target_ja=ja_text,
                            domain="general",
                            source_ref="jesc",
                            quality_score=0.7,
                            license="cc-by-4.0",
                        )
                    )
                if len(rows) >= max_rows:
                    break
    except Exception as exc:
        print(f"[jesc-direct] Extraction error: {exc}")

    print(f"[jesc-direct] Collected {len(rows)} pairs")
    return rows


# ── Source: Local file ───────────────────────────────────────────────────────

def collect_local(file_path: str, max_rows: int) -> list[TranslationRow]:
    """Import from a local CSV or JSONL file.

    CSV: must have columns source_en, target_ja (others optional).
    JSONL: each line is a JSON object with at least source_en and target_ja.
    """
    p = Path(file_path)
    if not p.exists():
        print(f"[local] File not found: {p}")
        return []

    rows: list[TranslationRow] = []
    if p.suffix == ".csv":
        with p.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if len(rows) >= max_rows:
                    break
                en = r.get("source_en", "").strip()
                ja = r.get("target_ja", "").strip()
                if en and ja:
                    rows.append(
                        TranslationRow(
                            id=r.get("id", TranslationRow.generate_id("local")),
                            source_en=en,
                            target_ja=ja,
                            domain=r.get("domain", "general"),
                            source_ref=f"local:{p.name}",
                            quality_score=float(r.get("quality_score", 0.9)),
                            license=r.get("license", "custom"),
                        )
                    )
    else:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if len(rows) >= max_rows:
                    break
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                en = obj.get("source_en", "").strip()
                ja = obj.get("target_ja", obj.get("reference_ja", "")).strip()
                if en and ja:
                    rows.append(
                        TranslationRow(
                            id=obj.get("id", TranslationRow.generate_id("local")),
                            source_en=en,
                            target_ja=ja,
                            domain=obj.get("domain", "general"),
                            source_ref=f"local:{p.name}",
                            quality_score=float(obj.get("quality_score", 0.9)),
                            license=obj.get("license", "custom"),
                        )
                    )

    print(f"[local] Collected {len(rows)} pairs from {p}")
    return rows


# ── Orchestrator ─────────────────────────────────────────────────────────────

SOURCE_REGISTRY: dict[str, any] = {
    "tatoeba": collect_tatoeba,
    "hf_tatoeba": collect_hf_tatoeba,
    "opus100": collect_opus100,
    "jesc": collect_jesc,
}


def run_collection(
    sources: list[str],
    max_per_source: int,
    local_file: str | None,
) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[TranslationRow] = []

    for src in sources:
        if src == "local":
            if not local_file:
                print("[warn] --local-file required when using 'local' source")
                continue
            rows = collect_local(local_file, max_per_source)
        elif src in SOURCE_REGISTRY:
            rows = SOURCE_REGISTRY[src](max_per_source)
        else:
            print(f"[warn] Unknown source: {src}")
            continue

        if rows:
            out = RAW_DIR / f"{src}_raw.jsonl"
            write_rows(rows, out)
            print(f"  -> Wrote {len(rows)} rows to {out}")
            all_rows.extend(rows)

    combined = RAW_DIR / "combined_raw.jsonl"
    write_rows(all_rows, combined)
    print(f"\nTotal: {len(all_rows)} pairs -> {combined}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect EN-JA parallel data")
    parser.add_argument(
        "--sources",
        default="tatoeba",
        help="Comma-separated list: tatoeba,hf_tatoeba,opus100,jesc,local",
    )
    parser.add_argument("--max-per-source", type=int, default=5000)
    parser.add_argument("--local-file", default=None)
    parser.add_argument(
        "--validate-only",
        default=None,
        help="Validate an existing JSONL file and exit",
    )
    args = parser.parse_args()

    if args.validate_only:
        valid, errors = validate_jsonl_file(args.validate_only)
        print(f"Valid rows: {valid}")
        if errors:
            print(f"Errors ({len(errors)}):")
            for e in errors:
                print(f"  {e}")
            sys.exit(1)
        print("All rows valid.")
        sys.exit(0)

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    run_collection(sources, args.max_per_source, args.local_file)


if __name__ == "__main__":
    main()
