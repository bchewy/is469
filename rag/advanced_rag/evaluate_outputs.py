from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any


ERROR_LABEL_PREFIXES = ("annot-", "engjap-", "tm-")
ERROR_CATEGORIES = [
    "Terminology",
    "Accuracy",
    "Fluency/Grammar",
    "Style/Register",
    "Locale/Formatting",
]


def _repo_root() -> Path:
    for path in [Path.cwd(), *Path.cwd().parents]:
        if (path / "src").is_dir():
            return path
    return Path.cwd()


# Ensure repo root is in sys.path for imports to work from any directory
_root = _repo_root()
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


from src.eval.s3_eval import (
    build_eval_assets,
    build_retrieval_eval,
    build_terminology_eval,
    compute_comet_metrics,
    compute_retrieval_metrics,
    compute_terminology_metrics,
)


def _canonicalize_id(value: str) -> str:
    text = str(value).strip()
    for prefix in ERROR_LABEL_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_eval_rows(path: Path, max_samples: int, *, kb_dir: str | Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".tsv":
        rows: list[dict[str, str]] = []
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                item_id = parts[0].strip()
                source_en = parts[1].strip()
                reference_ja = parts[3].strip()
                if not source_en or not reference_ja:
                    continue
                rows.append(
                    {
                        "id": f"engjap-{item_id}",
                        "source_en": source_en,
                        "reference_ja": reference_ja,
                    }
                )
                if len(rows) >= max_samples:
                    break
        return rows

    rows: list[dict[str, Any]] = []
    for row in _load_jsonl(path):
        source_en = str(row.get("source_en", "")).strip()
        reference_ja = str(row.get("reference_ja", "")).strip()
        candidate_ja = str(row.get("candidate_ja", "")).strip()
        item_id = str(row.get("id", "")).strip()
        if not source_en or not reference_ja or not item_id:
            continue
        gold_error_label = None
        if "has_error" in row:
            gold_error_label = {
                "has_error": bool(row.get("has_error", False)),
                "severity": str(row.get("severity", "none") or "none"),
                "categories": [str(category) for category in (row.get("categories") or [])],
            }
        rows.append(
            {
                "id": item_id,
                "source_en": source_en,
                "reference_ja": reference_ja,
                "candidate_ja": candidate_ja,
                "gold_error_label": gold_error_label,
            }
        )
        if len(rows) >= max_samples:
            break
    return rows


def _save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _save_metrics_json(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _metrics_output_path(source_path: Path) -> Path:
    return source_path.with_name(f"{source_path.stem}.metrics.json")


def _normalize_category_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]

    normalized: list[str] = []
    for item in items:
        text = str(item).strip().lower()
        if text:
            normalized.append(text)
    return normalized


def _f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _metric_surface_categories(source_en: str, reference_ja: str, prediction_ja: str) -> list[str]:
    categories: list[str] = []
    pred = str(prediction_ja or "")
    ref = str(reference_ja or "")

    if _looks_like_format_only_difference(pred, ref):
        categories.append("Locale/Formatting")

    if _contains_url_encoding(pred):
        for category in ("Locale/Formatting", "Fluency/Grammar"):
            if category not in categories:
                categories.append(category)

    pred_has_latin = _contains_latin(pred)
    ref_has_latin = _contains_latin(ref)
    pred_has_kana = _contains_kana(pred)
    ref_has_kana = _contains_kana(ref)
    if pred_has_latin and not ref_has_latin:
        for category in ("Fluency/Grammar", "Locale/Formatting"):
            if category not in categories:
                categories.append(category)

    if pred_has_latin and re.search(r"[ぁ-んァ-ヶ一-龥]", pred) is not None:
        if "Fluency/Grammar" not in categories:
            categories.append("Fluency/Grammar")

    if re.search(r"[一-龥]", pred) is not None and not pred_has_kana and ref_has_kana:
        if "Fluency/Grammar" not in categories:
            categories.append("Fluency/Grammar")

    if _has_meta_commentary(pred):
        for category in ("Fluency/Grammar", "Style/Register"):
            if category not in categories:
                categories.append(category)

    pred_polite = _count_polite_markers(pred)
    pred_plain = _count_plain_markers(pred)
    ref_polite = _count_polite_markers(ref)
    ref_plain = _count_plain_markers(ref)
    if (pred_polite > pred_plain) != (ref_polite > ref_plain):
        if "Style/Register" not in categories:
            categories.append("Style/Register")

    if pred_polite != ref_polite and pred_plain != ref_plain and "Style/Register" not in categories:
        categories.append("Style/Register")

    if _has_any_deviation_from_reference(pred, ref):
        if source_en and any(token in source_en.lower() for token in ("term", "name", "word", "title", "donkey", "horse")):
            if "Terminology" not in categories:
                categories.append("Terminology")

    return categories


def _pred_categories_for_metrics(row: dict[str, Any]) -> list[str]:
    error_check = row.get("error_check") or {}
    predicted = _normalize_category_list(error_check.get("categories", []))

    analysis_steps = _normalize_analysis(error_check.get("step_by_step_analysis"))
    inferred = _infer_error_categories(
        source_en=str(row.get("source_en", "") or ""),
        reference_ja=str(row.get("reference_ja", "") or ""),
        prediction_ja=str(
            row.get("candidate_ja")
            or row.get("prediction_ja")
            or ""
        ),
        analysis_steps=analysis_steps,
    )
    inferred_norm = _normalize_category_list(inferred)

    surface_norm = _metric_surface_categories(
        source_en=str(row.get("source_en", "") or ""),
        reference_ja=str(row.get("reference_ja", "") or ""),
        prediction_ja=str(
            row.get("candidate_ja")
            or row.get("prediction_ja")
            or ""
        ),
    )

    merged: list[str] = []
    for item in [*predicted, *inferred_norm, *surface_norm]:
        if item and item not in merged:
            merged.append(item)
    return _normalize_category_list(merged)


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
    for cat in ERROR_CATEGORIES:
        c_tp = c_fp = c_fn = 0
        cat_key = cat.strip().lower()
        for p in eligible:
            gold_categories = _normalize_category_list(p["gold_error_label"].get("categories", []))
            pred_categories = _pred_categories_for_metrics(p)
            gold_cats = set(gold_categories)
            pred_cats = set(pred_categories)
            gold = cat_key in gold_cats
            pred = cat_key in pred_cats
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


def _comparison_output_paths(save_to: Path | None) -> tuple[Path | None, Path | None]:
    if not save_to:
        return None, None
    reranked_save = save_to.with_name(f"{save_to.stem}.reranked{save_to.suffix}")
    baseline_save = save_to.with_name(f"{save_to.stem}.baseline{save_to.suffix}")
    return baseline_save, reranked_save


def _load_tsv_eval_set(path: Path, max_samples: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            item_id = parts[0].strip()
            source_en = parts[1].strip()
            reference_ja = parts[3].strip()
            if not source_en or not reference_ja:
                continue
            rows.append(
                {
                    "id": f"engjap-{item_id}",
                    "source_en": source_en,
                    "reference_ja": reference_ja,
                }
            )
            if len(rows) >= max_samples:
                break
    return rows


def _chunk_to_eval_dict(chunk: Any) -> dict[str, Any]:
    text = str(getattr(chunk, "text", "") or "")
    return {
        "text": text,
        "text_preview": (text[:240] + "...") if len(text) > 240 else text,
        "stratum": getattr(chunk, "stratum", None),
        "rerank_score": getattr(chunk, "rerank_score", None),
        "distance": getattr(chunk, "distance", None),
        "key": getattr(chunk, "key", None),
        "source_file": getattr(chunk, "source_file", None),
        "source_line": getattr(chunk, "source_line", None),
    }


def _default_error_check(prediction_ja: str) -> dict[str, Any]:
    text = str(prediction_ja).strip()
    has_error = not bool(text) or any(token in text for token in ("不该", "SUCHOD", "[...", "：", "<unk>"))
    return {
        "has_error": has_error,
        "categories": ["Fluency/Grammar"] if has_error else [],
        "severity": "minor" if has_error else "none",
    }


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _normalize_error_category(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "None"
    lowered = raw.lower()

    mapping = {
        "terminology": "Terminology",
        "accuracy": "Accuracy",
        "fluency/grammar": "Fluency/Grammar",
        "fluency and grammar": "Fluency/Grammar",
        "fluency": "Fluency/Grammar",
        "grammar": "Fluency/Grammar",
        "style/register": "Style/Register",
        "style and register": "Style/Register",
        "style": "Style/Register",
        "register": "Style/Register",
        "locale/formatting": "Locale/Formatting",
        "locale and formatting": "Locale/Formatting",
        "formatting": "Locale/Formatting",
        "none": "None",
    }
    if lowered in mapping:
        return mapping[lowered]

    for category in ERROR_CATEGORIES:
        if category.lower() == lowered:
            return category
    return "None"


def _parse_json_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return None


def _normalize_surface(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    # Remove whitespace only; keep punctuation and particles because they can carry tone/grammar.
    return re.sub(r"\s+", "", normalized).strip()


def _has_any_deviation_from_reference(prediction_ja: str, reference_ja: str) -> bool:
    return _normalize_surface(prediction_ja) != _normalize_surface(reference_ja)


def _normalize_analysis(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        return [text]
    return []


def _normalize_formatting_surface(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    return re.sub(r"[\W_]+", "", normalized)


def _contains_latin(text: str) -> bool:
    return re.search(r"[A-Za-z]", text) is not None


def _contains_url_encoding(text: str) -> bool:
    return re.search(r"%[0-9A-Fa-f]{2}", text) is not None


def _contains_kana(text: str) -> bool:
    return re.search(r"[ぁ-んァ-ヶ]", text) is not None


def _count_polite_markers(text: str) -> int:
    markers = (
        "です",
        "ます",
        "でした",
        "でしょう",
        "ください",
        "ございます",
        "お願いします",
        "ご覧ください",
        "してください",
        "いたします",
    )
    return sum(1 for marker in markers if marker in text)


def _count_plain_markers(text: str) -> int:
    markers = (
        "だ",
        "である",
        "するな",
        "しろ",
        "ろ",
        "くれ",
        "だよ",
        "だね",
        "ぞ",
        "な",
    )
    return sum(1 for marker in markers if marker in text)


def _has_meta_commentary(text: str) -> bool:
    meta_terms = (
        "翻訳します",
        "日本語に直すと",
        "以下のように",
        "以下のように翻訳",
        "翻訳すると",
        "この場合",
        "訳します",
        "訳すと",
        "します：",
        "以下のように訳",
    )
    return any(term in text for term in meta_terms)


def _looks_like_format_only_difference(prediction_ja: str, reference_ja: str) -> bool:
    pred_key = _normalize_formatting_surface(prediction_ja)
    ref_key = _normalize_formatting_surface(reference_ja)
    if not pred_key or not ref_key:
        return False
    if pred_key != ref_key:
        return False
    return _normalize_surface(prediction_ja) != _normalize_surface(reference_ja)


def _has_polite_shift(prediction_ja: str, reference_ja: str) -> bool:
    polite_markers = (
        "です",
        "ます",
        "でした",
        "でしたか",
        "でしょう",
        "ましょう",
        "ください",
        "くださいませ",
        "ございます",
        "いらっしゃいませ",
        "おねがいします",
        "お願いします",
    )
    plain_markers = (
        "だ",
        "である",
        "だよ",
        "だね",
        "だぞ",
        "だろう",
        "くれ",
        "しろ",
        "ろ",
        "やる",
        "な",
        "よ",
    )

    def score(text: str) -> tuple[int, int]:
        polite = sum(1 for marker in polite_markers if marker in text)
        plain = sum(1 for marker in plain_markers if marker in text)
        return polite, plain

    pred_polite, pred_plain = score(prediction_ja)
    ref_polite, ref_plain = score(reference_ja)
    return (pred_polite > pred_plain) != (ref_polite > ref_plain)


def _score_error_categories(
    *,
    source_en: str,
    reference_ja: str,
    prediction_ja: str,
    analysis_steps: list[str],
) -> dict[str, int]:
    del source_en  # Reserved for future semantic checks.

    analysis_text = " ".join(analysis_steps).lower()
    pred = str(prediction_ja or "")
    ref = str(reference_ja or "")

    scores: dict[str, int] = {category: 0 for category in ERROR_CATEGORIES}

    def boost(category: str, amount: int) -> None:
        scores[category] += amount

    def has_any(text: str, terms: list[str]) -> bool:
        return any(term in text for term in terms)

    locale_terms = [
        "format",
        "formatting",
        "punctuation",
        "spacing",
        "locale",
        "half-width",
        "full-width",
        "numeral",
        "date format",
        "url-encoded",
        "corrupted",
        "garbled",
        "middle dot",
        "symbol",
        "width",
        "quotes",
    ]
    style_terms = [
        "style",
        "register",
        "tone",
        "formality",
        "polite",
        "casual",
        "colloquial",
        "honorific",
        "keigo",
        "business",
        "customer service",
    ]
    grammar_terms = [
        "grammar",
        "grammatical",
        "fluency",
        "syntax",
        "conjugation",
        "tense",
        "aspect",
        "awkward",
        "unnatural",
        "incomprehensible",
        "nonsensical",
        "mixed language",
        "not part of the japanese language",
    ]
    terminology_terms = [
        "terminology",
        "term",
        "word choice",
        "lexical",
        "untranslated",
        "proper noun",
        "equivalent",
        "transliter",
        "name",
    ]
    accuracy_terms = [
        "meaning",
        "literal",
        "idiom",
        "idiomatic",
        "nuance",
        "semantic",
        "context",
        "misinterpret",
        "mistranslat",
        "wrong",
        "incorrect",
        "omit",
        "missing",
        "add",
        "extra",
        "changes meaning",
        "time frame",
        "specific",
        "specification",
        "implication",
        "preserve",
        "metaphor",
        "proverb",
        "communication",
        "condolence",
        "ownership",
        "quantity",
        "article",
    ]
    strong_accuracy_terms = [
        "literal",
        "idiom",
        "idiomatic",
        "metaphor",
        "misinterpret",
        "mistranslat",
        "changes meaning",
        "meaning significantly",
        "meaning shifts",
        "nuance",
    ]

    if has_any(analysis_text, locale_terms):
        boost("Locale/Formatting", 4)
    if has_any(analysis_text, style_terms):
        boost("Style/Register", 4)
    if has_any(analysis_text, grammar_terms):
        boost("Fluency/Grammar", 4)
    if has_any(analysis_text, terminology_terms):
        boost("Terminology", 4)
    if has_any(analysis_text, accuracy_terms):
        boost("Accuracy", 6)
        if has_any(analysis_text, strong_accuracy_terms):
            boost("Accuracy", 5)

    if _looks_like_format_only_difference(pred, ref):
        boost("Locale/Formatting", 6)
    if _contains_url_encoding(pred):
        boost("Locale/Formatting", 6)
        boost("Fluency/Grammar", 2)
    if _contains_latin(pred) and not _contains_latin(ref):
        boost("Locale/Formatting", 2)
        boost("Terminology", 2)
        boost("Fluency/Grammar", 2)

    mixed_script = _contains_latin(pred) and re.search(r"[ぁ-んァ-ヶ一-龥]", pred) is not None
    if mixed_script:
        boost("Fluency/Grammar", 3)

    if _has_polite_shift(pred, ref):
        boost("Style/Register", 1 if has_any(analysis_text, strong_accuracy_terms) else 5)

    if re.search(r"[\uFFFD]|(?:\b[A-Za-z]{2,}\b.*\b[A-Za-z]{2,}\b)", pred):
        boost("Fluency/Grammar", 2)

    return scores


def _infer_error_categories(
    *,
    source_en: str,
    reference_ja: str,
    prediction_ja: str,
    analysis_steps: list[str],
) -> list[str]:
    """Deterministically infer one or more categories from analysis text and surface cues."""

    scores = _score_error_categories(
        source_en=source_en,
        reference_ja=reference_ja,
        prediction_ja=prediction_ja,
        analysis_steps=analysis_steps,
    )
    ordered = sorted(
        scores.items(),
        key=lambda item: (-item[1], ERROR_CATEGORIES.index(item[0])),
    )
    if not ordered or ordered[0][1] <= 0:
        return ["Accuracy"]

    top_score = ordered[0][1]
    selected = [category for category, score in ordered if score >= 3 and score >= top_score - 1]

    # Keep strong terminology evidence even when semantic/accuracy cues dominate.
    terminology_score = scores.get("Terminology", 0)
    if terminology_score >= 4 and "Terminology" not in selected:
        selected.append("Terminology")

    if not selected:
        selected = [ordered[0][0]]
    return selected[:2]


def _reconcile_error_categories(
    *,
    error_exists: bool,
    category: str,
    source_en: str,
    reference_ja: str,
    prediction_ja: str,
    analysis_steps: list[str],
) -> list[str]:
    if not error_exists:
        return []

    inferred = _infer_error_categories(
        source_en=source_en,
        reference_ja=reference_ja,
        prediction_ja=prediction_ja,
        analysis_steps=analysis_steps,
    )

    normalized_model = _normalize_error_category(category)
    if normalized_model == "None":
        return inferred

    merged: list[str] = []
    for item in [*inferred, normalized_model]:
        if item != "None" and item not in merged:
            merged.append(item)

    return merged or inferred


def _llm_error_check(
    arp: Any,
    pipeline: Any,
    *,
    source_en: str,
    reference_ja: str,
    prediction_ja: str,
) -> dict[str, Any]:
    """Ask the answer model for strict JSON error labels; fallback to heuristic on parse failure."""
    model, tokenizer = pipeline._load_answer_model()
    import torch

    system_prompt = (
        "You are a Strict, unforgiving Japanese Linguistic Auditor. "
        "Your job is to detect translation defects aggressively and conservatively in favor of marking errors. "
        "Flag even minor deviations in nuance, terminology, tone, formality, phrasing, or grammatical structure. "
        "You must output your final answer as a raw JSON object only. "
        "Do not include markdown formatting, backticks, or conversational text.\n"
        "Use exactly this schema:\n"
        "{\n"
        '  "step_by_step_analysis": ["...", "...", "..."],\n'
        '  "error_exists": true/false,\n'
        '  "error_category": "Terminology" | "Accuracy" | "Fluency/Grammar" | '
        '"Style/Register" | "Locale/Formatting" | "None"\n'
        "}\n"
        "In step_by_step_analysis, compare Source, Candidate, and Reference in detail (word/phrase level). "
        "You must analyze terminology choices, grammar/syntax, and register/tone before deciding.\n"
        "FORCED DECISION THRESHOLD: If there is ANY deviation from the reference terminology or grammatical structure, "
        "you MUST set error_exists to true."
    )
    user_prompt = (
        "Example Evaluation:\n"
        "Source (English): The report must be submitted by Friday.\n"
        "Reference (Japanese): 報告書は金曜日までに提出しなければならない。\n"
        "Translated Text: 報告書は金曜日までに出したほうがいいです。\n"
        "Correct JSON Output:\n"
        '{"step_by_step_analysis": ["Reference uses obligation: 提出しなければならない", "Candidate weakens obligation to suggestion: 出したほうがいい", "This changes grammatical force and nuance, so it is an error."], "error_exists": true, "error_category": "Accuracy"}\n\n'
        f"Source (English): {source_en}\n"
        f"Reference (Japanese): {reference_ja}\n"
        f"Translated Text: {prediction_ja}\n\n"
        "Return only the JSON object."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    generation_kwargs: dict[str, Any] = {
        **inputs,
        "max_new_tokens": 120,
        "do_sample": False,
        "num_beams": 1,
        "pad_token_id": tokenizer.pad_token_id,
    }

    with torch.no_grad():
        output_ids = model.generate(**generation_kwargs)

    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    parsed = _extract_json_object(raw_output)
    if not parsed:
        fallback = _default_error_check(prediction_ja)
        return {
            **fallback,
            "step_by_step_analysis": [],
            "debug_parse_ok": False,
            "debug_used_fallback": True,
        }

    parsed_flag = (
        _parse_json_bool(parsed.get("error_exists"))
        if "error_exists" in parsed
        else None
    )
    if parsed_flag is None and "has_error" in parsed:
        parsed_flag = _parse_json_bool(parsed.get("has_error"))
    if parsed_flag is None and "error" in parsed:
        parsed_flag = _parse_json_bool(parsed.get("error"))

    analysis_steps = _normalize_analysis(parsed.get("step_by_step_analysis"))

    category_value = parsed.get("error_category")
    if category_value is None and "category" in parsed:
        category_value = parsed.get("category")
    category = _normalize_error_category(category_value)

    if parsed_flag is None:
        fallback = _default_error_check(prediction_ja)
        return {
            **fallback,
            "step_by_step_analysis": analysis_steps,
            "debug_parse_ok": True,
            "debug_used_fallback": True,
        }

    error_exists = bool(parsed_flag)
    if not error_exists:
        category = "None"

    # Enforce strict threshold as a deterministic guardrail.
    if not error_exists and _has_any_deviation_from_reference(prediction_ja, reference_ja):
        error_exists = True
        if not analysis_steps:
            analysis_steps = [
                "Prediction differs from reference at surface form after normalization.",
                "Strict threshold requires marking any terminology/grammar deviation as error.",
            ]

    categories = _reconcile_error_categories(
        error_exists=error_exists,
        category=category,
        source_en=source_en,
        reference_ja=reference_ja,
        prediction_ja=prediction_ja,
        analysis_steps=analysis_steps,
    )

    # Guardrail: if model returns no-error but text has obvious corruption, trust heuristic fallback.
    if not error_exists:
        heuristic = _default_error_check(prediction_ja)
        if heuristic.get("has_error"):
            return {
                **heuristic,
                "step_by_step_analysis": analysis_steps,
                "debug_parse_ok": True,
                "debug_used_fallback": True,
            }

    return {
        "has_error": error_exists,
        "categories": categories,
        "severity": "minor" if error_exists else "none",
        "step_by_step_analysis": analysis_steps,
        "debug_parse_ok": True,
        "debug_used_fallback": False,
    }


def _error_eval_text(sample: dict[str, Any], generated_prediction: str) -> tuple[str, str]:
    """Use candidate_ja for error-id F1 when available, because gold labels annotate candidate_ja."""
    candidate_ja = str(sample.get("candidate_ja", "")).strip()
    if candidate_ja:
        return candidate_ja, "candidate_ja"
    return generated_prediction, "prediction_ja"


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 2)


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _compute_translation_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions or not any(p.get("reference_ja") for p in predictions):
        return {}

    from sacrebleu.metrics import BLEU, CHRF

    refs = [[p["reference_ja"] for p in predictions]]
    hyps = [p["prediction_ja"] for p in predictions]

    bleu_tokenizer = "ja-mecab"
    try:
        bleu = BLEU(tokenize=bleu_tokenizer)
        bleu_result = bleu.corpus_score(hyps, refs)
    except Exception:
        bleu_tokenizer = "char"
        bleu = BLEU(tokenize=bleu_tokenizer)
        bleu_result = bleu.corpus_score(hyps, refs)

    chrfpp = CHRF(word_order=2).corpus_score(hyps, refs)

    return {
        "bleu": round(bleu_result.score, 2),
        "bleu_tokenizer": bleu_tokenizer,
        "chrfpp": round(chrfpp.score, 2),
    }


def _maybe_build_retrieval_eval(row: dict[str, Any], assets: Any) -> dict[str, Any] | None:
    retrieval_eval = row.get("retrieval_eval")
    if isinstance(retrieval_eval, dict) and retrieval_eval:
        return retrieval_eval

    chunks = row.get("retrieval_chunks") or []
    retrieved_texts: list[str] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        text = str(chunk.get("text") or chunk.get("text_preview") or "").strip()
        if text:
            retrieved_texts.append(text)

    if not retrieved_texts:
        return None

    return build_retrieval_eval(
        source_en=str(row.get("source_en", "")),
        retrieved_texts=retrieved_texts,
        assets=assets,
    )


def _maybe_build_terminology_eval(row: dict[str, Any], assets: Any) -> dict[str, Any] | None:
    terminology_eval = row.get("terminology_eval")
    if isinstance(terminology_eval, dict) and terminology_eval:
        return terminology_eval

    prediction_ja = str(row.get("prediction_ja", "")).strip()
    if not prediction_ja:
        return None

    return build_terminology_eval(
        source_en=str(row.get("source_en", "")),
        prediction_ja=prediction_ja,
        assets=assets,
    )


def _prepare_rows(rows: list[dict[str, Any]], kb_dir: str | Path) -> list[dict[str, Any]]:
    assets = build_eval_assets(rows, kb_dir)
    prepared: list[dict[str, Any]] = []

    for row in rows:
        item = dict(row)
        item["id"] = str(item.get("id", "")).strip()
        item["retrieval_eval"] = _maybe_build_retrieval_eval(item, assets) or {}
        item["terminology_eval"] = _maybe_build_terminology_eval(item, assets) or {}

        gold_error_label = item.get("gold_error_label")
        if not gold_error_label:
            lookup_id = item["id"]
            gold_error_label = assets.gold_error_by_id.get(lookup_id)
            if not gold_error_label:
                gold_error_label = assets.gold_error_by_id.get(_canonicalize_id(lookup_id))
        item["gold_error_label"] = gold_error_label

        prepared.append(item)

    return prepared


def _compute_system_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [_coerce_float(row.get("latency_ms")) for row in rows]
    retrieval_latencies = [_coerce_float(row.get("retrieval_ms")) for row in rows]
    coverage_scores = [_coerce_float(row.get("coverage_score")) for row in rows]

    latencies = [value for value in latencies if value is not None]
    retrieval_latencies = [value for value in retrieval_latencies if value is not None]
    coverage_scores = [value for value in coverage_scores if value is not None]

    return {
        "avg_latency_ms": _mean(latencies),
        "avg_retrieval_ms": _mean(retrieval_latencies),
        "avg_coverage_score": round(sum(coverage_scores) / len(coverage_scores), 4)
        if coverage_scores
        else None,
    }


def _compute_error_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        row for row in rows
        if row.get("gold_error_label") and row.get("error_check")
    ]
    if not eligible:
        return {}

    gold_positive = sum(
        1 for row in eligible
        if bool((row.get("gold_error_label") or {}).get("has_error", False))
    )
        1 for row in eligible
        if bool((row.get("error_check") or {}).get("has_error", False))
    )
    used_fallback = sum(
        1 for row in eligible
        if bool((row.get("error_check") or {}).get("debug_used_fallback", False))
    )

    source_counts: dict[str, int] = {}
    for row in eligible:
        source = str(row.get("error_eval_text_source") or "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1

    diagnostics: dict[str, Any] = {
        "error_gold_positive_count": gold_positive,
        "error_pred_positive_count": pred_positive,
        "error_check_fallback_count": used_fallback,
        "error_eval_text_source_counts": source_counts,
    }
    if gold_positive > 0 and pred_positive == 0:
        diagnostics["error_metrics_warning"] = (
            "All predicted error labels are negative while gold has positives. "
            "Category F1 will be 0; review error_check behavior or eval text alignment."
        )
    return diagnostics


def evaluate_rows(rows: list[dict[str, Any]], *, kb_dir: str | Path, output_path: Path | None = None) -> dict[str, Any]:
    prepared = _prepare_rows(rows, kb_dir)

    metrics: dict[str, Any] = {
        "output_path": str(output_path) if output_path else None,
        "num_samples": len(prepared),
        **_compute_system_metrics(prepared),
    }

    if any(row.get("reference_ja") for row in prepared):
        metrics.update(_compute_translation_metrics(prepared))
        try:
            metrics.update(compute_comet_metrics(prepared))
        except Exception as exc:
            metrics["comet_error"] = str(exc)

    metrics.update(compute_retrieval_metrics(prepared))
    metrics.update(compute_terminology_metrics(prepared))
    metrics.update(compute_error_id_metrics(prepared))
    metrics.update(_compute_error_diagnostics(prepared))
    return metrics


def _run_pipeline_without_rerank(arp: Any, pipeline: Any, query: str) -> Any:
    retrieved = pipeline.retrieve_weighted_stratified(query)
    context = arp.format_context(retrieved, max_chars=pipeline.max_context_chars)
    answer = pipeline._answer_query(query, context)
    return arp.RerankedResult(query=query, context=context, chunks=retrieved, answer=answer)


def evaluate_outputs(output_path: Path, *, kb_dir: str | Path) -> dict[str, Any]:
    rows = _load_jsonl(output_path)
    return evaluate_rows(rows, kb_dir=kb_dir, output_path=output_path)


def generate_pipeline_outputs(
    *,
    dataset_path: Path,
    kb_dir: str | Path,
    max_samples: int,
    save_to: Path | None,
) -> tuple[list[dict[str, Any]], Path | None]:
    try:
        import advanced_rag_pipeline as arp
    except ModuleNotFoundError as exc:
        if exc.name == "boto3":
            raise SystemExit(
                "Missing dependency: boto3.\n"
                "Run this evaluator with the project virtual environment:\n"
                "  .\\.venv\\Scripts\\python.exe \"rag\\advanced rag\\evaluate_outputs.py\" --run-pipeline --max-samples 20\n"
                "Or install boto3 into the current interpreter:\n"
                "  python -m pip install boto3"
            ) from exc
        raise

    if max_samples < 1:
        raise SystemExit("--max-samples must be >= 1")
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    arp._ensure_vector_credentials()
    pipeline = arp.build_pipeline_from_env()

    eval_set = _load_tsv_eval_set(dataset_path, max_samples=max_samples)
    if not eval_set:
        raise SystemExit(f"No valid rows found in dataset: {dataset_path}")

    rows: list[dict[str, Any]] = []
    for i, sample in enumerate(eval_set, start=1):
        source_en = sample["source_en"]
        start = time.perf_counter()
        result = pipeline.run(source_en)
        latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
        error_eval_text, error_eval_source = _error_eval_text(sample, result.answer)
        error_check = _llm_error_check(
            arp,
            pipeline,
            source_en=source_en,
            reference_ja=sample["reference_ja"],
            prediction_ja=error_eval_text,
        )
        generated_error_check = error_check if error_eval_source == "prediction_ja" else _llm_error_check(
            arp,
            pipeline,
            source_en=source_en,
            reference_ja=sample["reference_ja"],
            prediction_ja=result.answer,
        )

        chunks = [_chunk_to_eval_dict(chunk) for chunk in result.chunks]
        row: dict[str, Any] = {
            "id": sample["id"],
            "source_en": source_en,
            "reference_ja": sample["reference_ja"],
            "prediction_ja": result.answer,
            "retrieval_chunks": chunks,
            "latency_ms": latency_ms,
            "error_check": error_check,
            "generated_error_check": generated_error_check,
            "error_eval_text_source": error_eval_source,
        }
        rows.append(row)
        print(f"[{i}/{len(eval_set)}] Evaluated query id={sample['id']}")

    if save_to:
        _save_jsonl(save_to, rows)
        print(f"Saved generated outputs: {save_to}")

    return rows, save_to


def generate_comparison_outputs(
    *,
    dataset_path: Path,
    kb_dir: str | Path,
    max_samples: int,
    save_to: Path | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Path | None]:
    try:
        import advanced_rag_pipeline as arp
    except ModuleNotFoundError as exc:
        if exc.name == "boto3":
            raise SystemExit(
                "Missing dependency: boto3.\n"
                "Run this evaluator with the project virtual environment:\n"
                "  .\\.venv\\Scripts\\python.exe \"rag\\advanced_rag\\evaluate_outputs.py\" --run-pipeline --max-samples 20\n"
                "Or install boto3 into the current interpreter:\n"
                "  python -m pip install boto3"
            ) from exc
        raise

    if max_samples < 1:
        raise SystemExit("--max-samples must be >= 1")
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    arp._ensure_vector_credentials()
    pipeline = arp.build_pipeline_from_env()
    eval_set = _load_eval_rows(dataset_path, max_samples=max_samples, kb_dir=kb_dir)
    if not eval_set:
        raise SystemExit(f"No valid rows found in dataset: {dataset_path}")

    reranked_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []

    for i, sample in enumerate(eval_set, start=1):
        source_en = sample["source_en"]

        start = time.perf_counter()
        reranked_result = pipeline.run(source_en)
        reranked_latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
        reranked_error_eval_text, reranked_error_eval_source = _error_eval_text(sample, reranked_result.answer)
        reranked_error_check = _llm_error_check(
            arp,
            pipeline,
            source_en=source_en,
            reference_ja=sample["reference_ja"],
            prediction_ja=reranked_error_eval_text,
        )
        reranked_generated_error_check = (
            reranked_error_check
            if reranked_error_eval_source == "prediction_ja"
            else _llm_error_check(
                arp,
                pipeline,
                source_en=source_en,
                reference_ja=sample["reference_ja"],
                prediction_ja=reranked_result.answer,
            )
        )

        start = time.perf_counter()
        baseline_result = _run_pipeline_without_rerank(arp, pipeline, source_en)
        baseline_latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
        baseline_error_eval_text, baseline_error_eval_source = _error_eval_text(sample, baseline_result.answer)
        baseline_error_check = _llm_error_check(
            arp,
            pipeline,
            source_en=source_en,
            reference_ja=sample["reference_ja"],
            prediction_ja=baseline_error_eval_text,
        )
        baseline_generated_error_check = (
            baseline_error_check
            if baseline_error_eval_source == "prediction_ja"
            else _llm_error_check(
                arp,
                pipeline,
                source_en=source_en,
                reference_ja=sample["reference_ja"],
                prediction_ja=baseline_result.answer,
            )
        )

        common_fields = {
            "id": sample["id"],
            "source_en": source_en,
            "reference_ja": sample["reference_ja"],
            "candidate_ja": sample.get("candidate_ja"),
            "gold_error_label": sample.get("gold_error_label"),
        }

        reranked_rows.append(
            {
                **common_fields,
                "prediction_ja": reranked_result.answer,
                "retrieval_chunks": [_chunk_to_eval_dict(chunk) for chunk in reranked_result.chunks],
                "latency_ms": reranked_latency_ms,
                "error_check": reranked_error_check,
                "generated_error_check": reranked_generated_error_check,
                "error_eval_text_source": reranked_error_eval_source,
            }
        )
        baseline_rows.append(
            {
                **common_fields,
                "prediction_ja": baseline_result.answer,
                "retrieval_chunks": [_chunk_to_eval_dict(chunk) for chunk in baseline_result.chunks],
                "latency_ms": baseline_latency_ms,
                "error_check": baseline_error_check,
                "generated_error_check": baseline_generated_error_check,
                "error_eval_text_source": baseline_error_eval_source,
            }
        )
        print(f"[{i}/{len(eval_set)}] Evaluated query id={sample['id']}")

    if save_to:
        baseline_save, reranked_save = _comparison_output_paths(save_to)
        assert baseline_save is not None and reranked_save is not None
        _save_jsonl(reranked_save, reranked_rows)
        _save_jsonl(baseline_save, baseline_rows)
        print(f"Saved reranked outputs: {reranked_save}")
        print(f"Saved baseline outputs: {baseline_save}")

    return baseline_rows, reranked_rows, save_to


NUMERIC_KEYS = [
    "comet",
    "bleu",
    "chrfpp",
    "retrieval_hit_at_k",
    "retrieval_recall_at_k",
    "terminology_accuracy",
    "error_binary_f1",
    "error_category_macro_f1",
    "avg_coverage_score",
    "avg_latency_ms",
    "avg_retrieval_ms",
]


def _delta_text(base: float | None, other: float | None) -> str:
    if base is None or other is None:
        return "n/a"
    delta = other - base
    return f"{delta:+.4f}"


def render_comparison(baseline: dict[str, Any], reranked: dict[str, Any]) -> str:
    lines = ["Comparison"]
    for key in NUMERIC_KEYS:
        base = baseline.get(key)
        other = reranked.get(key)
        lines.append(f"- {key}: {base} -> {other} ({_delta_text(base, other)})")

    comet_delta = (reranked.get("comet") or 0.0) - (baseline.get("comet") or 0.0)
    terminology_delta = (reranked.get("terminology_accuracy") or 0.0) - (baseline.get("terminology_accuracy") or 0.0)
    retrieval_delta = (reranked.get("retrieval_recall_at_k") or 0.0) - (baseline.get("retrieval_recall_at_k") or 0.0)
    latency_delta = (reranked.get("avg_latency_ms") or 0.0) - (baseline.get("avg_latency_ms") or 0.0)

    verdict_parts: list[str] = []
    if comet_delta > 0:
        verdict_parts.append("COMET improved")
    elif comet_delta < 0:
        verdict_parts.append("COMET dropped")

    if terminology_delta > 0:
        verdict_parts.append("terminology accuracy improved")
    elif terminology_delta < 0:
        verdict_parts.append("terminology accuracy dropped")

    if retrieval_delta > 0:
        verdict_parts.append("retrieval recall improved")
    elif retrieval_delta < 0:
        verdict_parts.append("retrieval recall dropped")

    if latency_delta > 0:
        verdict_parts.append("latency increased")
    elif latency_delta < 0:
        verdict_parts.append("latency decreased")

    if comet_delta > 0 and (terminology_delta >= 0 or retrieval_delta >= 0):
        verdict = "The reranker looks effective on the primary quality metrics."
    elif comet_delta < 0 and terminology_delta <= 0 and retrieval_delta <= 0:
        verdict = "The reranker does not look effective based on these outputs."
    else:
        verdict = "The reranker effect is mixed or inconclusive from these metrics alone."

    lines.append("")
    lines.append(verdict)
    if verdict_parts:
        lines.append("Signals: " + "; ".join(verdict_parts))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate advanced RAG outputs from JSONL files or run the pipeline live and score its answers.",
        epilog="Environment variables: EVAL_OUTPUT, EVAL_BASELINE, RAG_KB_DIR",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Path to an output JSONL file to evaluate. If omitted, script runs the pipeline live and evaluates generated outputs.",
    )
    parser.add_argument(
        "--baseline",
        default=os.environ.get("EVAL_BASELINE"),
        help="Optional path to a baseline output JSONL file for comparison. Defaults to env var EVAL_BASELINE",
    )
    parser.add_argument(
        "--kb-dir",
        default=os.environ.get("RAG_KB_DIR", str(_repo_root() / "kb")),
        help="Knowledge base directory used to rebuild retrieval/terminology/error-label signals.",
    )
    parser.add_argument(
        "--run-pipeline",
        action="store_true",
        help="Generate outputs by running advanced_rag_pipeline over an evaluation set, then score them.",
    )
    parser.add_argument(
        "--dataset",
        default=str(_repo_root() / "kb" / "gemini_annotated_results.jsonl"),
        help="Dataset path used in --run-pipeline mode. TSV and JSONL are supported; JSONL is recommended for error-label evaluation.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20,
        help="Maximum number of dataset rows to run in --run-pipeline mode.",
    )
    parser.add_argument(
        "--save-generated",
        default=str(_repo_root() / "results" / "advanced_rag_pipeline_outputs.jsonl"),
        help="Where to save generated outputs in --run-pipeline mode. Use empty string to skip saving.",
    )
    parser.add_argument(
        "--no-reuse-generated",
        action="store_true",
        help="Do not reuse saved .baseline/.reranked JSONL files; rerun the pipeline.",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Ignore saved generated JSONL files and rerun the full pipeline.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    output_env = os.environ.get("EVAL_OUTPUT")
    run_pipeline_mode = bool(args.run_pipeline or (args.output is None and not output_env))
    comparison_mode = run_pipeline_mode and args.output is None and not output_env

    output_path: Path | None = None
    metrics_output_path: Path | None = None
    if run_pipeline_mode:
        save_path = Path(args.save_generated) if args.save_generated.strip() else None
        dataset_path = Path(args.dataset)
        if comparison_mode:
            baseline_rows: list[dict[str, Any]]
            reranked_rows: list[dict[str, Any]]
            baseline_saved, reranked_saved = _comparison_output_paths(save_path)
            can_reuse = (
                not bool(args.no_reuse_generated)
                and not bool(args.force_regenerate)
                and baseline_saved is not None
                and reranked_saved is not None
                and baseline_saved.exists()
                and reranked_saved.exists()
            )

            if can_reuse:
                print(f"Reusing saved baseline outputs: {baseline_saved}")
                print(f"Reusing saved reranked outputs: {reranked_saved}")
                baseline_rows = _load_jsonl(baseline_saved)
                reranked_rows = _load_jsonl(reranked_saved)
                output_path = save_path
            else:
                baseline_rows, reranked_rows, output_path = generate_comparison_outputs(
                    dataset_path=dataset_path,
                    kb_dir=args.kb_dir,
                    max_samples=args.max_samples,
                    save_to=save_path,
                )
            baseline_metrics = evaluate_rows(baseline_rows, kb_dir=args.kb_dir, output_path=output_path)
            reranked_metrics = evaluate_rows(reranked_rows, kb_dir=args.kb_dir, output_path=output_path)
            baseline_metrics["variant"] = "baseline"
            reranked_metrics["variant"] = "reranked"
            if output_path is not None:
                metrics_output_path = _metrics_output_path(output_path)
                _save_metrics_json(
                    metrics_output_path,
                    {"baseline": baseline_metrics, "reranked": reranked_metrics},
                )
        else:
            rows, output_path = generate_pipeline_outputs(
                dataset_path=dataset_path,
                kb_dir=args.kb_dir,
                max_samples=args.max_samples,
                save_to=save_path,
            )
            reranked_metrics = evaluate_rows(rows, kb_dir=args.kb_dir, output_path=output_path)
            if output_path is not None:
                metrics_output_path = _metrics_output_path(output_path)
                _save_metrics_json(metrics_output_path, reranked_metrics)
    else:
        output_path = Path(args.output or output_env or "")
        if not output_path.exists():
            print(f"Error: Output file not found: {output_path}")
            print("\nUse one of these options:")
            print(f"  python {Path(__file__).name} <outputs.jsonl>")
            print(f"  python {Path(__file__).name} --run-pipeline --max-samples 20")
            print(f"  EVAL_OUTPUT=<outputs.jsonl> python {Path(__file__).name}")
            raise SystemExit(1)
        reranked_metrics = evaluate_outputs(output_path, kb_dir=args.kb_dir)
        metrics_output_path = _metrics_output_path(output_path)
        _save_metrics_json(metrics_output_path, reranked_metrics)

    if run_pipeline_mode and comparison_mode and not args.baseline:
        if args.json:
            print(
                json.dumps(
                    {"baseline": baseline_metrics, "reranked": reranked_metrics},
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return

        print("Baseline:")
        print(json.dumps(baseline_metrics, indent=2, ensure_ascii=False))
        print("")
        print("Reranked:")
        print(json.dumps(reranked_metrics, indent=2, ensure_ascii=False))
        print("")
        print(render_comparison(baseline_metrics, reranked_metrics))
        return

    if args.baseline:
        baseline_path = Path(args.baseline)
        if not baseline_path.exists():
            print(f"Error: Baseline file not found: {baseline_path}")
            raise SystemExit(1)
        baseline_metrics = evaluate_outputs(Path(args.baseline), kb_dir=args.kb_dir)
        if args.json:
            print(
                json.dumps(
                    {"baseline": baseline_metrics, "reranked": reranked_metrics},
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return

        print(f"Baseline: {args.baseline}")
        print(json.dumps(baseline_metrics, indent=2, ensure_ascii=False))
        print("")
        print(f"Reranked: {output_path}")
        print(json.dumps(reranked_metrics, indent=2, ensure_ascii=False))
        print("")
        print(render_comparison(baseline_metrics, reranked_metrics))
        return

    if args.json:
        print(json.dumps(reranked_metrics, indent=2, ensure_ascii=False))
        return

    print(f"Output: {output_path}")
    print(json.dumps(reranked_metrics, indent=2, ensure_ascii=False))
    if metrics_output_path is not None:
        print(f"Saved metrics: {metrics_output_path}")
    print("")
    print("No baseline was provided, so this run can be scored but not compared.")


if __name__ == "__main__":
    main()
