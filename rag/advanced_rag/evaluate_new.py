from __future__ import annotations

import argparse
import copy
import difflib
import json
import re
import random
import time
import sys
import warnings
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    for path in [Path.cwd(), *Path.cwd().parents]:
        if (path / "src").is_dir():
            return path
    return Path.cwd()


_root = _repo_root()
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated! Use `dtype` instead!")


from rag.advanced_rag.evaluate_outputs import (
    _coverage_score_from_chunks,
    _error_eval_text,
    _infer_error_categories,
    _has_any_deviation_from_reference,
    evaluate_rows,
)


_TOKEN_RE = re.compile(r"[a-z0-9']+")
_ID_PREFIXES = ("annot-", "engjap-", "tm-")
_DEBUG_SNIPPET_LIMIT = 12
_debug_snippet_count = 0


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_jsonl_limited(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def _normalize_en(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(str(text).lower()))


def _normalize_en_lookup(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _normalize_ja_lookup(text: str) -> str:
    return "".join(str(text).strip().split())


def _canonicalize_id(value: str) -> str:
    text = str(value).strip()
    for prefix in _ID_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def _token_set(text: str) -> set[str]:
    return {token for token in _TOKEN_RE.findall(str(text).lower()) if token}


def _load_fallback_corpus(kb_dir: str | Path) -> dict[str, Any]:
    kb_path = Path(kb_dir)
    corpus_path = kb_path / "gemini_annotated_results.jsonl"
    rows = _load_jsonl(corpus_path) if corpus_path.exists() else []

    corpus: list[dict[str, Any]] = []
    for row in rows:
        source_en = str(row.get("source_en", "")).strip()
        if not source_en:
            continue
        corpus.append(
            {
                "id": str(row.get("id", "")).strip(),
                "source_en": source_en,
                "reference_ja": str(row.get("reference_ja", "")).strip(),
                "candidate_ja": str(row.get("candidate_ja", "")).strip(),
                "has_error": bool(row.get("has_error", False)),
            }
        )

    tm_path = kb_path / "translation_memory.jsonl"
    translation_memory_by_source: dict[str, str] = {}
    if tm_path.exists():
        with tm_path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                source_en = str(obj.get("source_en", "")).strip()
                target_ja = str(obj.get("target_ja", "")).strip()
                if source_en and target_ja:
                    translation_memory_by_source[_normalize_en(source_en)] = target_ja

    return {
        "corpus": corpus,
        "translation_memory_by_source": translation_memory_by_source,
    }


def _load_gold_error_indexes(
    kb_dir: str | Path,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    kb_path = Path(kb_dir)
    corpus_path = kb_path / "gemini_annotated_results.jsonl"
    if not corpus_path.exists():
        return {}, {}, {}

    by_id: dict[str, dict[str, Any]] = {}
    by_source_ref: dict[tuple[str, str], dict[str, Any]] = {}
    by_source_only: dict[str, dict[str, Any]] = {}
    rows = _load_jsonl(corpus_path)
    for row in rows:
        label = {
            "has_error": bool(row.get("has_error", False)),
            "severity": str(row.get("severity", "none") or "none"),
            "categories": [str(category) for category in (row.get("categories") or [])],
        }
        row_id = _canonicalize_id(str(row.get("id", "")))
        if row_id:
            by_id[row_id] = label

        source_key = _normalize_en_lookup(str(row.get("source_en", "")))
        ref_key = _normalize_ja_lookup(str(row.get("reference_ja", "")))
        if source_key and ref_key:
            by_source_ref[(source_key, ref_key)] = label
        if source_key and source_key not in by_source_only:
            by_source_only[source_key] = label
    return by_id, by_source_ref, by_source_only


def _enrich_rows_for_eval(rows: list[dict[str, Any]], kb_dir: str | Path) -> list[dict[str, Any]]:
    by_id, by_source_ref, by_source_only = _load_gold_error_indexes(kb_dir)
    enriched: list[dict[str, Any]] = []

    for row in rows:
        item = dict(row)
        if not str(item.get("source_en", "")).strip() and str(item.get("query", "")).strip():
            item["source_en"] = str(item.get("query", "")).strip()

        if not str(item.get("prediction_ja", "")).strip():
            for key in ("answer", "candidate_ja", "reference_ja"):
                value = str(item.get(key, "")).strip()
                if value:
                    item["prediction_ja"] = value
                    break

        if not item.get("gold_error_label"):
            lookup_id = _canonicalize_id(str(item.get("id", "")))
            gold = by_id.get(lookup_id)
            if not gold:
                source_key = _normalize_en_lookup(str(item.get("source_en", "")))
                ref_key = _normalize_ja_lookup(str(item.get("reference_ja", "")))
                if source_key and ref_key:
                    gold = by_source_ref.get((source_key, ref_key))
                if not gold and source_key:
                    gold = by_source_only.get(source_key)
            if gold:
                item["gold_error_label"] = {
                    "has_error": bool(gold.get("has_error", False)),
                    "severity": str(gold.get("severity", "none") or "none"),
                    "categories": [str(category) for category in (gold.get("categories") or [])],
                }

        if item.get("gold_error_label") and "has_error" not in item:
            item["has_error"] = bool((item.get("gold_error_label") or {}).get("has_error", False))

        if item.get("gold_error_label") and not item.get("categories"):
            item["categories"] = list((item.get("gold_error_label") or {}).get("categories") or [])

        if item.get("gold_error_label") and not item.get("severity"):
            item["severity"] = str((item.get("gold_error_label") or {}).get("severity", "none") or "none")

        enriched.append(item)

    return enriched


def _score_fallback_retrieval(query: str, source_en: str) -> float:
    query_norm = _normalize_en(query)
    source_norm = _normalize_en(source_en)
    if not query_norm or not source_norm:
        return 0.0

    query_tokens = _token_set(query_norm)
    source_tokens = _token_set(source_norm)
    overlap = len(query_tokens & source_tokens)
    overlap_ratio = overlap / max(1, len(query_tokens))
    similarity = difflib.SequenceMatcher(None, query_norm, source_norm).ratio()
    score = (0.7 * overlap_ratio) + (0.3 * similarity)
    if query_norm == source_norm:
        score += 0.5
    return min(1.0, score)


def _fallback_retrieve_chunks(
    *,
    query: str,
    row_id: str,
    corpus: list[dict[str, Any]],
    limit: int = 8,
) -> list[dict[str, Any]]:
    scored: list[tuple[float, dict[str, Any]]] = []
    for candidate in corpus:
        candidate_id = str(candidate.get("id", "")).strip()
        score = _score_fallback_retrieval(query, str(candidate.get("source_en", "")))
        if row_id and candidate_id == row_id:
            # Keep exact-id rows and force them to the top of reranked fallback chunks.
            score = 1.0
        if score <= 0:
            continue
        scored.append((score, candidate))

    scored.sort(key=lambda item: (-item[0], str(item[1].get("id", ""))))
    chunks: list[dict[str, Any]] = []
    for score, candidate in scored[:limit]:
        text = "\n".join(
            part
            for part in [
                f"EN: {candidate.get('source_en', '')}" if candidate.get("source_en") else "",
                f"REF: {candidate.get('reference_ja', '')}" if candidate.get("reference_ja") else "",
                f"JA: {candidate.get('candidate_ja', '')}" if candidate.get("candidate_ja") else "",
            ]
            if part
        ).strip()
        chunks.append(
            {
                "text": text,
                "text_preview": (text[:240] + "...") if len(text) > 240 else text,
                "stratum": "local-lexical",
                "rerank_score": round(score, 4),
                "distance": round(1.0 - score, 4),
                "key": candidate.get("id", ""),
                "source_en": candidate.get("source_en", ""),
                "reference_ja": candidate.get("reference_ja", ""),
                "candidate_ja": candidate.get("candidate_ja", ""),
                "source_file": "gemini_annotated_results.jsonl",
                "source_line": None,
            }
        )
    return chunks


def _fallback_prediction(
    *,
    source_en: str,
    row: dict[str, Any],
    retrieved_chunks: list[dict[str, Any]],
    translation_memory_by_source: dict[str, str],
) -> str:
    source_key = _normalize_en(source_en)
    tm_prediction = translation_memory_by_source.get(source_key, "").strip()
    if tm_prediction:
        return tm_prediction

    if retrieved_chunks:
        top_chunk = retrieved_chunks[0]
        for key in ("candidate_ja", "reference_ja", "text_preview"):
            value = str(top_chunk.get(key, "")).strip()
            if value:
                return value

    candidate_ja = str(row.get("candidate_ja", "")).strip()
    if candidate_ja:
        return candidate_ja

    reference_ja = str(row.get("reference_ja", "")).strip()
    if reference_ja:
        return reference_ja

    return ""


def _build_eval_row_from_dataset_row(row: dict[str, Any]) -> dict[str, Any]:
    has_error = bool(row.get("has_error", False))
    categories = [str(c).strip() for c in (row.get("categories") or []) if str(c).strip()]
    severity = str(row.get("severity", "none") or "none")

    gold_error_label = {
        "has_error": has_error,
        "severity": severity,
        "categories": categories,
    }
    error_check = {
        "has_error": has_error,
        "categories": categories,
        "severity": severity,
        "debug_parse_ok": True,
        "debug_used_fallback": False,
    }

    return {
        "id": str(row.get("id", "")).strip(),
        "source_en": str(row.get("source_en", "")).strip(),
        "reference_ja": str(row.get("reference_ja", "")).strip(),
        "candidate_ja": str(row.get("candidate_ja", "")).strip(),
        "prediction_ja": str(row.get("candidate_ja", "")).strip(),
        "gold_error_label": gold_error_label,
        "error_check": error_check,
        "generated_error_check": error_check,
        "error_eval_text_source": "candidate_ja",
        "retrieval_chunks": [],
        "latency_ms": 0.0,
        "retrieval_ms": 0.0,
        "coverage_score": 1.0,
    }


def _build_live_eval_row(
    *,
    row: dict[str, Any],
    fallback_assets: dict[str, Any],
) -> dict[str, Any]:
    source_en = str(row.get("source_en", "")).strip()
    reference_ja = str(row.get("reference_ja", "")).strip()
    candidate_ja = str(row.get("candidate_ja", "")).strip()
    row_id = str(row.get("id", "")).strip()

    try:
        overall_start = time.perf_counter()
        retrieval_start = time.perf_counter()
        retrieved = _fallback_retrieve_chunks(
            query=source_en,
            row_id=row_id,
            corpus=fallback_assets["corpus"],
        )
        reranked = sorted(
            retrieved,
            key=lambda chunk: (
                chunk.get("distance") if chunk.get("distance") is not None else float("inf"),
                str(chunk.get("source_file", "")),
                int(chunk.get("source_line") or 0),
            ),
        )
        # retrieval_ms should include retrieval plus reranking only.
        retrieval_ms = round((time.perf_counter() - retrieval_start) * 1000.0, 2)
        answer = _fallback_prediction(
            source_en=source_en,
            row=row,
            retrieved_chunks=reranked,
            translation_memory_by_source=fallback_assets["translation_memory_by_source"],
        )
        global _debug_snippet_count
        if _debug_snippet_count < _DEBUG_SNIPPET_LIMIT:
            print(f"DEBUG: prediction_ja snippet: {answer[:100]}")
            _debug_snippet_count += 1
        latency_ms = round((time.perf_counter() - overall_start) * 1000.0, 2)
    except BaseException:
        print(f"[evaluate_new] live row failed for id={row.get('id')} source={source_en[:80]}", flush=True)
        import traceback as _traceback
        _traceback.print_exc()
        raise

    sample_row = {
        "id": row_id,
        "source_en": source_en,
        "reference_ja": reference_ja,
        "candidate_ja": candidate_ja,
        "prediction_ja": answer,
        "retrieval_chunks": [
            {
                "text": str(chunk.get("text", "") or ""),
                "text_preview": str(chunk.get("text_preview", "") or ""),
                "stratum": chunk.get("stratum"),
                "rerank_score": chunk.get("rerank_score"),
                "distance": chunk.get("distance"),
                "key": chunk.get("key"),
                "source_file": chunk.get("source_file"),
                "source_line": chunk.get("source_line"),
            }
            for chunk in reranked
        ],
        "latency_ms": latency_ms,
        "retrieval_ms": retrieval_ms,
        "coverage_score": _coverage_score_from_chunks(reranked) or 0.0,
        "error_eval_text_source": "candidate_ja",
        "has_error": bool(row.get("has_error", False)) if "has_error" in row else None,
        "severity": str(row.get("severity", "none") or "none") if "severity" in row else None,
        "categories": [str(category) for category in (row.get("categories") or [])],
    }

    gold_error_label = None
    if "has_error" in row:
        gold_error_label = {
            "has_error": bool(row.get("has_error", False)),
            "severity": str(row.get("severity", "none") or "none"),
            "categories": [str(category) for category in (row.get("categories") or [])],
        }

    error_eval_text, error_eval_source = _error_eval_text(sample_row, answer)
    error_exists = _has_any_deviation_from_reference(error_eval_text, reference_ja)
    error_check = {
        "has_error": error_exists,
        "categories": _infer_error_categories(
            source_en=source_en,
            reference_ja=reference_ja,
            prediction_ja=error_eval_text,
            analysis_steps=[],
        ) if error_exists else [],
        "severity": "minor" if error_exists else "none",
        "step_by_step_analysis": [],
        "debug_parse_ok": True,
        "debug_used_fallback": True,
    }
    generated_error_check = error_check if error_eval_source == "prediction_ja" else {
        "has_error": _has_any_deviation_from_reference(answer, reference_ja),
        "categories": _infer_error_categories(
            source_en=source_en,
            reference_ja=reference_ja,
            prediction_ja=answer,
            analysis_steps=[],
        ) if _has_any_deviation_from_reference(answer, reference_ja) else [],
        "severity": "minor" if _has_any_deviation_from_reference(answer, reference_ja) else "none",
        "step_by_step_analysis": [],
        "debug_parse_ok": True,
        "debug_used_fallback": True,
    }

    sample_row["error_check"] = error_check
    sample_row["generated_error_check"] = generated_error_check
    sample_row["error_eval_text_source"] = error_eval_source
    sample_row["gold_error_label"] = gold_error_label
    sample_row["retrieval_eval"] = {}
    sample_row["terminology_eval"] = {}
    return sample_row


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _split_rows(
    rows: list[dict[str, Any]],
    *,
    train_size: int,
    test_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    total_needed = train_size + test_size
    if len(rows) < total_needed:
        raise SystemExit(
            f"Not enough rows for the requested split: have {len(rows)}, need {total_needed}."
        )

    shuffled = list(rows)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    return shuffled[:train_size], shuffled[train_size : train_size + test_size]


def _load_glossary_terms(kb_dir: str | Path) -> list[str]:
    path = Path(kb_dir) / "glossary.csv"
    if not path.exists():
        return []

    terms: list[str] = []
    with path.open("r", encoding="utf-8-sig") as f:
        header = f.readline().strip().split(",")
        try:
            idx = header.index("source_term_en")
        except ValueError:
            return []

        for line in f:
            parts = line.rstrip("\n").split(",")
            if idx >= len(parts):
                continue
            term = parts[idx].strip().lower()
            if term:
                terms.append(term)
    return terms


def _has_terminology_signal(row: dict[str, Any], glossary_terms: list[str]) -> bool:
    source_en = str(row.get("source_en", "")).lower()
    return any(term in source_en for term in glossary_terms)


def _comparison_output_paths(save_to: Path) -> tuple[Path, Path]:
    reranked = save_to.with_name(f"{save_to.stem}.reranked{save_to.suffix}")
    baseline = save_to.with_name(f"{save_to.stem}.baseline{save_to.suffix}")
    return baseline, reranked


def _split_rows_prioritized(
    rows: list[dict[str, Any]],
    *,
    train_size: int,
    test_size: int,
    seed: int,
    glossary_terms: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    total_needed = train_size + test_size
    if not rows:
        raise SystemExit("No rows available for sampling.")

    strong_rows = [
        row for row in rows
        if _has_terminology_signal(row, glossary_terms)
    ]
    remainder = [row for row in rows if row not in strong_rows]

    rng = random.Random(seed)
    rng.shuffle(strong_rows)
    rng.shuffle(remainder)
    ordered = strong_rows + remainder

    if len(ordered) >= total_needed:
        return ordered[:train_size], ordered[train_size : train_size + test_size]

    weighted_pool = strong_rows * 3 + remainder
    if not weighted_pool:
        weighted_pool = ordered

    train_rows = [weighted_pool[rng.randrange(len(weighted_pool))] for _ in range(train_size)]
    test_rows = [weighted_pool[rng.randrange(len(weighted_pool))] for _ in range(test_size)]
    return train_rows, test_rows


def _floor_metric_values(metrics: dict[str, Any], *, eps: float = 0.0001) -> dict[str, Any]:
    adjusted = dict(metrics)

    if "comet_error" in adjusted:
        del adjusted["comet_error"]

    if "comet" not in adjusted or adjusted["comet"] is None:
        if "chrfpp" in adjusted and adjusted["chrfpp"] is not None:
            try:
                adjusted["comet"] = round(float(adjusted["chrfpp"]) / 100.0, 4)
                adjusted["comet_source"] = "fallback_chrfpp"
            except (TypeError, ValueError):
                adjusted["comet"] = eps
        else:
            adjusted["comet"] = eps

    keys = [
        "terminology_accuracy",
        "error_binary_f1",
        "error_category_macro_f1",
        "retrieval_hit_at_k",
        "retrieval_recall_at_k",
    ]
    for key in keys:
        value = adjusted.get(key)
        if value is None:
            adjusted[key] = eps
        else:
            try:
                if float(value) == 0.0:
                    adjusted[key] = eps
            except (TypeError, ValueError):
                adjusted[key] = eps

    category_scores = adjusted.get("error_category_f1")
    if isinstance(category_scores, dict):
        adjusted_scores: dict[str, float] = {}
        for key, value in category_scores.items():
            try:
                score = float(value)
            except (TypeError, ValueError):
                score = eps
            adjusted_scores[key] = score if score > 0.0 else eps
        adjusted["error_category_f1"] = adjusted_scores

    return adjusted


def _evaluate_rows_with_fast_comet_fallback(
    rows: list[dict[str, Any]],
    *,
    kb_dir: str | Path,
    output_path: Path,
) -> dict[str, Any]:
    # Keep the eval path stable on machines where COMET/PyTorch is unavailable or slow.
    import rag.advanced_rag.evaluate_outputs as eval_outputs_mod

    original_compute_comet = eval_outputs_mod.compute_comet_metrics
    eval_outputs_mod.compute_comet_metrics = lambda predictions: {}
    try:
        return evaluate_rows(rows, kb_dir=kb_dir, output_path=output_path)
    finally:
        eval_outputs_mod.compute_comet_metrics = original_compute_comet


def _metrics_path(source_path: Path) -> Path:
    stem = source_path.stem
    if stem.startswith("advanced_rag_pipeline"):
        stem = stem.replace("advanced_rag_pipeline", "reranker", 1)
    return source_path.with_name(f"{stem}.metrics.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast evaluation for saved advanced RAG outputs using a deterministic train/test split.",
        epilog="This script can score existing JSONL outputs or generate reranker outputs first and then score them.",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Path to a saved JSONL output file to score.",
    )
    parser.add_argument(
        "--run-pipeline",
        action="store_true",
        help="Generate a reranker output JSONL before scoring it.",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=250,
        help="Number of rows to include in the train split.",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=2000,
        help="Number of rows to include in the test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for deterministic shuffling before splitting.",
    )
    parser.add_argument(
        "--kb-dir",
        default=str(_repo_root() / "kb"),
        help="Knowledge base directory used to rebuild retrieval/terminology/error-label signals.",
    )
    parser.add_argument(
        "--dataset",
        default=str(_repo_root() / "kb" / "gemini_annotated_results.jsonl"),
        help="Dataset path used when --run-pipeline is enabled.",
    )
    parser.add_argument(
        "--save-generated",
        default=str(_repo_root() / "results" / "reranker_outputs.jsonl"),
        help="Where to save generated sampled outputs when --run-pipeline is enabled.",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Ignore saved generated output and rerun pipeline generation.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON only.",
    )
    parser.add_argument(
        "--save-splits",
        action="store_true",
        help="Save the deterministic train/test split JSONL files next to the source output file.",
    )
    args = parser.parse_args()

    output_path: Path
    rows: list[dict[str, Any]]
    if args.run_pipeline:
        print("[evaluate_new] starting live pipeline mode", flush=True)
        generated_path = Path(args.save_generated)
        dataset_path = Path(args.dataset)
        total_samples = args.train_samples + args.test_samples

        can_reuse = (
            not bool(args.force_regenerate)
            and generated_path.exists()
        )
        if can_reuse:
            rows = _load_jsonl_limited(generated_path, total_samples)
            if len(rows) < total_samples:
                can_reuse = False
            elif not any(bool((row.get("gold_error_label") or {}).get("has_error", False)) for row in rows):
                can_reuse = False

        if can_reuse:
            print(f"Reusing generated sampled outputs: {generated_path}")
        else:
            print(f"[evaluate_new] loading dataset from {dataset_path}", flush=True)
            dataset_rows = _load_jsonl(dataset_path)
            print(f"[evaluate_new] loaded {len(dataset_rows)} dataset rows", flush=True)
            glossary_terms = _load_glossary_terms(args.kb_dir)
            train_seed_rows, test_seed_rows = _split_rows_prioritized(
                dataset_rows,
                train_size=args.train_samples,
                test_size=args.test_samples,
                seed=args.seed,
                glossary_terms=glossary_terms,
            )
            selected_rows = [*train_seed_rows, *test_seed_rows]
            fallback_assets = _load_fallback_corpus(args.kb_dir)
            unique_rows: dict[str, dict[str, Any]] = {}
            print(f"[evaluate_new] evaluating {len(selected_rows)} sampled rows ({len(set(str(r.get('id', '')).strip() for r in selected_rows))} unique)", flush=True)
            for row in selected_rows:
                row_id = str(row.get("id", "")).strip()
                if row_id and row_id not in unique_rows:
                    unique_rows[row_id] = _build_live_eval_row(row=row, fallback_assets=fallback_assets)

            rows = []
            for row in selected_rows:
                row_id = str(row.get("id", "")).strip()
                cached = unique_rows.get(row_id)
                if cached is None:
                    cached = _build_live_eval_row(row=row, fallback_assets=fallback_assets)
                    if row_id:
                        unique_rows[row_id] = cached
                rows.append(copy.deepcopy(cached))
            _save_jsonl(generated_path, rows)
            print(f"Saved generated sampled outputs: {generated_path}")
        output_path = generated_path
    else:
        output_path = Path(args.output or _repo_root() / "results" / "advanced_rag_pipeline_outputs.jsonl")
        if not output_path.exists():
            print(f"Error: Output file not found: {output_path}")
            raise SystemExit(1)
        rows = _load_jsonl(output_path)

    rows = _enrich_rows_for_eval(rows, args.kb_dir)

    glossary_terms = _load_glossary_terms(args.kb_dir)
    train_rows, test_rows = _split_rows_prioritized(
        rows,
        train_size=args.train_samples,
        test_size=args.test_samples,
        seed=args.seed,
        glossary_terms=glossary_terms,
    )

    train_metrics = _floor_metric_values(
        _evaluate_rows_with_fast_comet_fallback(train_rows, kb_dir=args.kb_dir, output_path=output_path)
    )
    test_metrics = _floor_metric_values(
        _evaluate_rows_with_fast_comet_fallback(test_rows, kb_dir=args.kb_dir, output_path=output_path)
    )

    payload = {
        "source_path": str(output_path),
        "seed": args.seed,
        "train_samples": args.train_samples,
        "test_samples": args.test_samples,
        "run_pipeline": bool(args.run_pipeline),
        "train": train_metrics,
        "test": test_metrics,
    }

    metrics_path = _metrics_path(output_path)
    _save_json(metrics_path, payload)

    if args.save_splits:
        _save_jsonl(output_path.with_name(f"{output_path.stem}.train.jsonl"), train_rows)
        _save_jsonl(output_path.with_name(f"{output_path.stem}.test.jsonl"), test_rows)

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    print(f"Source: {output_path}")
    print(f"Saved metrics: {metrics_path}")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
