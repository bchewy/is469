from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ERROR_LABEL_CATEGORIES = [
    "Terminology",
    "Accuracy",
    "Fluency/Grammar",
    "Style/Register",
    "Locale/Formatting",
]


def _normalize_en(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _normalize_ja(text: str) -> str:
    return "".join(text.strip().split())


def _canonicalize_id(value: str) -> str:
    text = str(value).strip()
    for prefix in ("annot-", "engjap-", "tm-"):
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


_EN_TOKEN_RE = re.compile(r"[a-z0-9']+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "for",
    "from", "has", "he", "her", "here", "his", "i", "if", "in", "is", "it", "its",
    "let", "may", "me", "my", "of", "on", "or", "our", "please", "she", "that",
    "the", "their", "them", "there", "they", "this", "to", "was", "we", "will",
    "with", "you", "your",
}


def _tokenize_en(text: str) -> set[str]:
    return {tok for tok in _EN_TOKEN_RE.findall(text.lower()) if tok not in _STOPWORDS}


def _extract_candidate_english_texts(retrieved_texts: list[str]) -> list[str]:
    candidates: list[str] = []
    for text in retrieved_texts:
        for line in text.splitlines():
            stripped = line.strip()
            if "EN:" in stripped:
                candidates.append(stripped.split("EN:", 1)[1].strip())
            elif stripped:
                candidates.append(stripped)
    return candidates


def _best_source_overlap(source_en: str, retrieved_texts: list[str]) -> float:
    source_tokens = _tokenize_en(source_en)
    if not source_tokens:
        return 0.0

    best = 0.0
    for candidate in _extract_candidate_english_texts(retrieved_texts):
        cand_tokens = _tokenize_en(candidate)
        if not cand_tokens:
            continue
        overlap = len(source_tokens & cand_tokens) / len(source_tokens)
        if overlap > best:
            best = overlap
    return best


@dataclass
class GlossaryEntry:
    source_term_en: str
    approved_ja: str
    forbidden_variants: list[str]


@dataclass
class EvalAssets:
    glossary_entries: list[GlossaryEntry]
    translation_memory_by_source: dict[str, dict[str, str]]
    exact_parallel_sources: set[str]
    gold_error_by_id: dict[str, dict[str, Any]]


def _load_glossary(kb_dir: Path) -> list[GlossaryEntry]:
    path = kb_dir / "glossary.csv"
    if not path.exists():
        return []

    entries: list[GlossaryEntry] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            source_term_en = str(row.get("source_term_en", "")).strip()
            approved_ja = str(row.get("approved_ja", "")).strip()
            forbidden_raw = str(row.get("forbidden_variants", "")).strip()
            forbidden = [x.strip() for x in forbidden_raw.split("|") if x.strip()]
            if source_term_en and approved_ja:
                entries.append(
                    GlossaryEntry(
                        source_term_en=source_term_en,
                        approved_ja=approved_ja,
                        forbidden_variants=forbidden,
                    )
                )
    return entries


def _load_translation_memory(kb_dir: Path, normalized_sources: set[str]) -> dict[str, dict[str, str]]:
    path = kb_dir / "translation_memory.jsonl"
    if not path.exists():
        return {}

    out: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            source_en = str(obj.get("source_en", "")).strip()
            target_ja = str(obj.get("target_ja", "")).strip()
            key = _normalize_en(source_en)
            if key in normalized_sources and source_en and target_ja:
                out[key] = {"source_en": source_en, "target_ja": target_ja}
    return out


def _load_exact_parallel_sources(kb_dir: Path, normalized_sources: set[str]) -> set[str]:
    path = kb_dir / "eng-jap.tsv"
    if not path.exists():
        return set()

    found: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            key = _normalize_en(parts[1])
            if key in normalized_sources:
                found.add(key)
    return found


def _load_gold_error_labels(kb_dir: Path) -> dict[str, dict[str, Any]]:
    path = kb_dir / "gemini_annotated_results.jsonl"
    if not path.exists():
        return {}

    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            row_id = _canonicalize_id(obj.get("id", ""))
            if not row_id:
                continue
            out[row_id] = {
                "has_error": bool(obj.get("has_error", False)),
                "severity": str(obj.get("severity", "none") or "none"),
                "categories": [
                    c for c in (str(x).strip() for x in obj.get("categories", []))
                    if c in ERROR_LABEL_CATEGORIES
                ],
                "rationale": str(obj.get("rationale", "")).strip(),
            }
    return out


def build_eval_assets(rows: list[dict[str, Any]], kb_dir: str | Path) -> EvalAssets:
    kb_path = Path(kb_dir)
    normalized_sources = {
        _normalize_en(str(row.get("source_en", "")))
        for row in rows
        if str(row.get("source_en", "")).strip()
    }
    return EvalAssets(
        glossary_entries=_load_glossary(kb_path),
        translation_memory_by_source=_load_translation_memory(kb_path, normalized_sources),
        exact_parallel_sources=_load_exact_parallel_sources(kb_path, normalized_sources),
        gold_error_by_id=_load_gold_error_labels(kb_path),
    )


def build_retrieval_eval(
    *,
    source_en: str,
    retrieved_texts: list[str],
    assets: EvalAssets,
) -> dict[str, Any]:
    norm_source = _normalize_en(source_en)
    combined = "\n".join(retrieved_texts)
    combined_lower = combined.lower()

    glossary_targets = [
        entry
        for entry in assets.glossary_entries
        if entry.source_term_en.lower() in norm_source
    ]

    expected_targets: list[dict[str, Any]] = []
    for entry in glossary_targets:
        matched = (
            entry.source_term_en.lower() in combined_lower
            or entry.approved_ja in combined
            or any(v in combined for v in entry.forbidden_variants)
        )
        expected_targets.append(
            {
                "kind": "glossary",
                "key": entry.source_term_en,
                "matched": matched,
            }
        )

    tm_entry = assets.translation_memory_by_source.get(norm_source)
    if tm_entry:
        matched = (
            tm_entry["source_en"] in combined
            or tm_entry["target_ja"] in combined
            or _best_source_overlap(source_en, retrieved_texts) >= 0.6
        )
        expected_targets.append(
            {
                "kind": "translation_memory",
                "key": tm_entry["source_en"],
                "matched": matched,
            }
        )

    if norm_source in assets.exact_parallel_sources:
        overlap = _best_source_overlap(source_en, retrieved_texts)
        matched = source_en in combined or overlap >= 0.6
        expected_targets.append(
            {
                "kind": "parallel_memory",
                "key": source_en,
                "matched": matched,
                "overlap": round(overlap, 4),
            }
        )

    matched_count = sum(1 for t in expected_targets if t["matched"])
    expected_count = len(expected_targets)
    return {
        "expected_target_count": expected_count,
        "matched_target_count": matched_count,
        "hit_at_k": bool(expected_count and matched_count),
        "recall": (matched_count / expected_count) if expected_count else None,
        "matched_kinds": sorted({t["kind"] for t in expected_targets if t["matched"]}),
        "expected_kinds": sorted({t["kind"] for t in expected_targets}),
    }


def build_terminology_eval(
    *,
    source_en: str,
    prediction_ja: str,
    assets: EvalAssets,
) -> dict[str, Any]:
    norm_source = _normalize_en(source_en)
    prediction_norm = _normalize_ja(prediction_ja)

    matched_terms = [
        entry for entry in assets.glossary_entries if entry.source_term_en.lower() in norm_source
    ]
    if not matched_terms:
        return {
            "term_count": 0,
            "correct_term_count": 0,
            "accuracy": None,
        }

    correct = 0
    for entry in matched_terms:
        approved_ok = _normalize_ja(entry.approved_ja) in prediction_norm
        forbidden_hit = any(_normalize_ja(v) in prediction_norm for v in entry.forbidden_variants)
        if approved_ok and not forbidden_hit:
            correct += 1

    return {
        "term_count": len(matched_terms),
        "correct_term_count": correct,
        "accuracy": correct / len(matched_terms) if matched_terms else None,
    }


def compute_retrieval_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    eval_rows = [p.get("retrieval_eval") or {} for p in predictions]
    eligible = [r for r in eval_rows if r.get("expected_target_count", 0) > 0]
    if not eligible:
        return {
            "retrieval_eval_samples": 0,
            "retrieval_hit_at_k": None,
            "retrieval_recall_at_k": None,
        }

    hits = sum(1 for r in eligible if r.get("hit_at_k"))
    matched = sum(int(r.get("matched_target_count", 0)) for r in eligible)
    expected = sum(int(r.get("expected_target_count", 0)) for r in eligible)
    return {
        "retrieval_eval_samples": len(eligible),
        "retrieval_hit_at_k": round(hits / len(eligible), 4),
        "retrieval_recall_at_k": round((matched / expected) if expected else 0.0, 4),
    }


def compute_terminology_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    eval_rows = [p.get("terminology_eval") or {} for p in predictions]
    eligible = [r for r in eval_rows if r.get("term_count", 0) > 0]
    if not eligible:
        return {
            "terminology_eval_samples": 0,
            "terminology_accuracy": None,
        }

    correct = sum(int(r.get("correct_term_count", 0)) for r in eligible)
    total = sum(int(r.get("term_count", 0)) for r in eligible)
    return {
        "terminology_eval_samples": len(eligible),
        "terminology_term_total": total,
        "terminology_correct_terms": correct,
        "terminology_accuracy": round((correct / total) if total else 0.0, 4),
    }


def _f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_error_id_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        p for p in predictions
        if p.get("gold_error_label") and p.get("error_check")
    ]
    if not eligible:
        return {
            "error_id_eval_samples": 0,
            "error_binary_f1": None,
            "error_category_macro_f1": None,
        }

    tp = fp = fn = 0
    for p in eligible:
        gold = bool(p["gold_error_label"].get("has_error", False))
        pred = bool(p["error_check"].get("has_error", False))
        if gold and pred:
            tp += 1
        elif pred and not gold:
            fp += 1
        elif gold and not pred:
            fn += 1

    category_scores: dict[str, float] = {}
    for cat in ERROR_LABEL_CATEGORIES:
        c_tp = c_fp = c_fn = 0
        for p in eligible:
            gold_cats = set(p["gold_error_label"].get("categories", []))
            pred_cats = set(p["error_check"].get("categories", []))
            gold = cat in gold_cats
            pred = cat in pred_cats
            if gold and pred:
                c_tp += 1
            elif pred and not gold:
                c_fp += 1
            elif gold and not pred:
                c_fn += 1
        category_scores[cat] = round(_f1(c_tp, c_fp, c_fn), 4)

    macro_f1 = sum(category_scores.values()) / len(category_scores)
    return {
        "error_id_eval_samples": len(eligible),
        "error_binary_f1": round(_f1(tp, fp, fn), 4),
        "error_category_macro_f1": round(macro_f1, 4),
        "error_category_f1": category_scores,
    }


def compute_comet_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions or not any(p.get("reference_ja") for p in predictions):
        return {}

    import torch
    from comet import download_model, load_from_checkpoint

    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    samples = [
        {
            "src": p["source_en"],
            "mt": p["prediction_ja"],
            "ref": p["reference_ja"],
        }
        for p in predictions
        if p.get("reference_ja")
    ]
    outputs = model.predict(
        samples,
        batch_size=8,
        gpus=1 if torch.cuda.is_available() else 0,
        progress_bar=False,
        num_workers=0,
    )
    return {
        "comet": round(float(outputs.system_score), 4),
    }
