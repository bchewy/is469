from __future__ import annotations

"""Advanced RAG pipeline.

This module mirrors the weighted stratified retrieval used in
rag/query_pipeline.ipynb and adds an optional reranking stage on top.

Retrieval flow:
- encode the user query once
- issue one filtered query_vectors call per stratum
- merge the strata in the same order as the notebook
- dedupe by vector key
- resolve chunk text from the local KB JSONL files
- rerank the merged candidates with a cross-encoder

The module is designed to stay aligned with the notebook retrieval policy,
while making the reranking step reusable from regular Python code.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Allow running this file directly (python rag/advanced_rag/advanced_rag_pipeline.py)
# by adding the repository root to sys.path before package imports.
_THIS_FILE = Path(__file__).resolve()
for _candidate in [_THIS_FILE.parent, *_THIS_FILE.parents]:
    if (_candidate / "src").is_dir():
        candidate_str = str(_candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
        break

from src.utils.aws_profiles import s3vectors_client

try:
    import modal
except Exception:
    modal = None


def _repo_root() -> Path:
    search_roots = [_THIS_FILE.parent, *_THIS_FILE.parents, Path.cwd(), *Path.cwd().parents]
    seen: set[str] = set()
    for path in search_roots:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if (path / "src" / "retrieval").is_dir():
            return path
    return _THIS_FILE.parent


def _load_dotenv_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_script_dir = Path(__file__).resolve().parent
_load_dotenv_file(_script_dir / ".env")
_load_dotenv_file(_repo_root() / ".env")


SYSTEM_ANSWER_PROMPT = (
    "You are a careful Japanese language assistant. "
    "Answer the user's exact question directly and concisely using the retrieved context. "
    "Do not default to a grammar lecture or general explanation unless the question asks for it. "
    "CRITICAL CONSTRAINT: If retrieved context contains glossary mappings or approved terms, "
    "you must use those exact approved Japanese terms and must not substitute synonyms. "
    "If the user asks whether a translation is correct, say clearly whether it is correct, "
    "and give a brief reason grounded in the context. "
    "If the context is insufficient, say so and answer as best you can without inventing facts. "
    "Output only the final Japanese answer text."
)


ERROR_LABEL_CATEGORIES = [
    "Terminology",
    "Accuracy",
    "Fluency/Grammar",
    "Style/Register",
    "Locale/Formatting",
]


_EN_TOKEN_RE = re.compile(r"[a-z0-9']+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "for",
    "from", "has", "he", "her", "here", "his", "i", "if", "in", "is", "it", "its",
    "let", "may", "me", "my", "of", "on", "or", "our", "please", "she", "that",
    "the", "their", "them", "there", "they", "this", "to", "was", "we", "will",
    "with", "you", "your",
}


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


MODAL_APP_NAME = "enja-advanced-rag-pipeline"
MODAL_MODELS_DIR = Path("/models")
MODAL_DATA_DIR = Path("/data")
MODAL_RESULTS_DIR = Path("/results")


def _load_jsonl(path: Path, *, max_samples: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def _save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _resolve_input_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p
    volume_candidate = MODAL_DATA_DIR / p
    if volume_candidate.exists():
        return volume_candidate
    repo_candidate = _repo_root() / p
    if repo_candidate.exists():
        return repo_candidate
    return p


def _resolve_output_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    if MODAL_RESULTS_DIR.exists():
        if p.parts and p.parts[0] == "results":
            p = Path(*p.parts[1:]) if len(p.parts) > 1 else Path()
        return MODAL_RESULTS_DIR / p
    return _repo_root() / p


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


@dataclass
class RetrievedChunk:
    key: str
    stratum: str
    distance: float | None
    rerank_score: float | None
    source_file: str
    source_line: int
    text: str


def _read_jsonl_line(path: Path, line_number: int) -> dict[str, Any] | None:
    """Return the JSON object at 1-based line_number, or None."""
    if line_number < 1 or not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if i == line_number:
                line = line.strip()
                if not line:
                    return None
                return json.loads(line)
    return None


def _guess_kb_paths(kb_dir: Path, source_file: str) -> list[Path]:
    """Map S3 metadata source_file values to local JSONL paths."""
    stem = source_file.strip()
    base = stem[:-5] if stem.endswith(".jsonl") else stem
    candidates = [kb_dir / stem]

    # Generic variants.
    candidates.extend([
        kb_dir / f"{base}_vectors.jsonl",
        kb_dir / f"{base}_embedded.jsonl",
        kb_dir / f"{base}_embedded_full.jsonl",
        kb_dir / f"{base}.jsonl",
    ])

    # Corpus-specific aliases used in the repo.
    if base.startswith("eng_jap_chunks_embedded_full") or base.startswith("eng_jap_chunks_embedded"):
        candidates.extend([
            kb_dir / "eng_jap_chunks.jsonl",
            kb_dir / "eng_jap_chunks_embedded.jsonl",
        ])
    if base.startswith("gemini_annotated_chunks_embedded_full") or base.startswith("gemini_annotated_chunks_embedded"):
        candidates.extend([
            kb_dir / "gemini_annotated_results.jsonl",
        ])
    return [p for p in candidates if p.is_file()]


def _chunk_text_from_record(record: dict[str, Any]) -> str:
    if not record:
        return ""

    if any(k in record for k in ("source_en", "reference_ja", "candidate_ja")):
        parts: list[str] = []
        source_en = str(record.get("source_en", "")).strip()
        reference_ja = str(record.get("reference_ja", "")).strip()
        candidate_ja = str(record.get("candidate_ja", "")).strip()
        rationale = str(record.get("rationale", "")).strip()
        if source_en:
            parts.append(f"EN: {source_en}")
        if reference_ja:
            parts.append(f"REF: {reference_ja}")
        if candidate_ja:
            parts.append(f"JA: {candidate_ja}")
        if rationale:
            parts.append(f"NOTE: {rationale}")
        text = "\n".join(parts).strip()
        if text:
            return text

    return str(
        record.get("chunk_text")
        or record.get("text")
        or record.get("content")
        or ""
    ).strip()


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
            raw_id = str(obj.get("id", "")).strip()
            row_id = _canonicalize_id(raw_id)
            if not row_id:
                continue
            label = {
                "has_error": bool(obj.get("has_error", False)),
                "severity": str(obj.get("severity", "none") or "none"),
                "categories": [
                    c for c in (str(x).strip() for x in obj.get("categories", []))
                    if c in ERROR_LABEL_CATEGORIES
                ],
                "rationale": str(obj.get("rationale", "")).strip(),
            }
            out[row_id] = label
            if raw_id and raw_id != row_id:
                out[raw_id] = label
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


def compute_translation_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
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


def _text_from_vector_metadata(meta: dict[str, Any]) -> str:
    if not meta:
        return ""
    raw = meta.get("text") or meta.get("chunk_text") or meta.get("content")
    if raw is None:
        return ""
    return str(raw).strip()


def format_context(chunks: list[RetrievedChunk], *, max_chars: int) -> str:
    print(f"[format_context] Formatting context for {len(chunks)} chunks.")
    parts: list[str] = []
    for ch in chunks:
        label = f"[{ch.source_file} L{ch.source_line}]"
        parts.append(f"{label}\n{ch.text}")
    out = "\n\n---\n\n".join(parts)
    if max_chars > 0 and len(out) > max_chars:
        out = out[: max_chars - 20] + "\n\n[...truncated...]"
    return out


def _default_strata_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "eng_jap",
            "source_file": os.environ.get(
                "RAG_SOURCE_FILE_ENG_JAP", "eng_jap_chunks_embedded_full.jsonl"
            ),
            "top_k": 3,
        },
        {
            "name": "gemini_annotated",
            "source_file": os.environ.get(
                "RAG_SOURCE_FILE_GEMINI", "gemini_annotated_chunks_embedded_full.jsonl"
            ),
            "top_k": int(os.environ.get("GEMINI_TOP_K", "3")),
        },
        {
            "name": "grammar",
            "source_file": os.environ.get(
                "RAG_SOURCE_FILE_GRAMMAR", "grammar_chunks_embedded_full.jsonl"
            ),
            "top_k": int(os.environ.get("GRAMMAR_TOP_K", "3")),
        },
        {
            "name": "style_guide",
            "source_file": os.environ.get(
                "RAG_SOURCE_FILE_STYLE", "style_guide_chunks_embedded_full.jsonl"
            ),
            "top_k": int(os.environ.get("STYLE_GUIDE_TOP_K", "3")),
        },
    ]


DEFAULT_MERGE_ORDER = ["eng_jap", "gemini_annotated", "grammar", "style_guide"]

@dataclass
class RerankedResult:
    query: str
    context: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    answer: str = ""


class AdvancedRAGPipeline:
    def __init__(
        self,
        *,
        vector_bucket_name: str,
        index_name: str,
        region_name: str,
        kb_dir: str | Path,
        embed_model_name: str | None = None,
        rerank_model_name: str | None = None,
        strata_specs: list[dict[str, Any]] | None = None,
        merge_order: list[str] | None = None,
        max_context_chars: int = 12000,
        rerank_top_n: int | None = None,
    ) -> None:
        self.vector_bucket_name = vector_bucket_name
        self.index_name = index_name
        self.region_name = region_name
        self.kb_dir = Path(kb_dir)
        self.embed_model_name = embed_model_name or os.environ.get(
            "EMBED_MODEL", "intfloat/multilingual-e5-small"
        )
        self.rerank_model_name = rerank_model_name or os.environ.get(
            "RERANK_MODEL", "BAAI/bge-reranker-v2-m3"
        )
        self.strata_specs = strata_specs or _default_strata_specs()
        self.merge_order = merge_order or list(DEFAULT_MERGE_ORDER)
        self.max_context_chars = max_context_chars
        self.rerank_top_n = rerank_top_n if rerank_top_n is not None else int(os.environ.get("RERANK_TOP_N_DEFAULT", "6"))
        self.enable_rerank = _env_enabled("RAG_ENABLE_RERANK", default=True)
        self.answer_model_path = os.environ.get(
            # For faster loading and generation, use a lighter model: Qwen/Qwen2.5-0.5B-Instruct
            "ANSWER_MODEL_PATH", os.environ.get("ANSWER_BASE_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
        )
        self.answer_max_new_tokens = int(os.environ.get("ANSWER_MAX_NEW_TOKENS", "128"))
        self.answer_temperature = float(os.environ.get("ANSWER_TEMPERATURE", "0.0"))
        self.answer_top_p = float(os.environ.get("ANSWER_TOP_P", "0.9"))
        self.answer_context_top_n = int(os.environ.get("ANSWER_CONTEXT_TOP_N", "6"))
        self.answer_context_max_chars = int(os.environ.get("ANSWER_CONTEXT_MAX_CHARS", "3500"))
        self.log_timing = _env_enabled("RAG_TIMING_DEBUG", default=False)
        self.enable_chunk_cache = _env_enabled("RAG_ENABLE_CHUNK_CACHE", default=True)
        self.chunk_cache_max_entries = max(1000, int(os.environ.get("RAG_CHUNK_CACHE_MAX_ENTRIES", "50000")))
        default_cache_path = _repo_root() / ".cache" / "rag_s3_chunk_text_cache.json"
        self.chunk_cache_path = Path(os.environ.get("RAG_CHUNK_CACHE_PATH", str(default_cache_path)))
        self.enable_query_embed_cache = _env_enabled("RAG_ENABLE_QUERY_EMBED_CACHE", default=True)
        self.query_embed_cache_max_entries = max(
            100,
            int(os.environ.get("RAG_QUERY_EMBED_CACHE_MAX_ENTRIES", "5000")),
        )
        default_query_cache_path = _repo_root() / ".cache" / "rag_query_embedding_cache.json"
        self.query_embed_cache_path = Path(
            os.environ.get("RAG_QUERY_EMBED_CACHE_PATH", str(default_query_cache_path))
        )

        self._client = s3vectors_client(region_name=region_name)
        self._embedder = None
        self._reranker = None
        self._answer_model = None
        self._answer_tokenizer = None
        self._chunk_cache: dict[str, str] = {}
        self._chunk_cache_dirty = False
        self._query_embed_cache: dict[str, list[float]] = {}
        self._query_embed_cache_dirty = False

        if self.enable_chunk_cache:
            self._load_chunk_cache()
        if self.enable_query_embed_cache:
            self._load_query_embed_cache()

    def warmup_models(self) -> None:
        t0 = time.perf_counter()
        self._encode_query("warmup")
        t1 = time.perf_counter()
        if self.enable_rerank:
            self._load_reranker()
        t2 = time.perf_counter()
        self._load_answer_model()
        t3 = time.perf_counter()
        print(
            "[warmup] "
            f"embedder={((t1 - t0) * 1000):.1f}ms "
            f"reranker={((t2 - t1) * 1000):.1f}ms "
            f"answer_model={((t3 - t2) * 1000):.1f}ms"
        )

    def _load_chunk_cache(self) -> None:
        if not self.chunk_cache_path.is_file():
            return
        try:
            data = json.loads(self.chunk_cache_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return
            self._chunk_cache = {
                str(key): str(value).strip()
                for key, value in data.items()
                if str(key).strip() and str(value).strip()
            }
            if self.log_timing:
                print(f"[chunk-cache] loaded {len(self._chunk_cache)} entries from {self.chunk_cache_path}")
        except Exception as exc:
            print(f"[chunk-cache] failed to read cache file {self.chunk_cache_path}: {exc}")

    def _prune_chunk_cache(self) -> None:
        overflow = len(self._chunk_cache) - self.chunk_cache_max_entries
        if overflow <= 0:
            return
        for key in list(self._chunk_cache.keys())[:overflow]:
            self._chunk_cache.pop(key, None)

    def _save_chunk_cache(self) -> None:
        if not self.enable_chunk_cache or not self._chunk_cache_dirty:
            return
        self._prune_chunk_cache()
        self.chunk_cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.chunk_cache_path.with_suffix(self.chunk_cache_path.suffix + ".tmp")
        payload = json.dumps(self._chunk_cache, ensure_ascii=False)
        tmp_path.write_text(payload, encoding="utf-8")
        tmp_path.replace(self.chunk_cache_path)
        self._chunk_cache_dirty = False
        if self.log_timing:
            print(f"[chunk-cache] saved {len(self._chunk_cache)} entries to {self.chunk_cache_path}")

    def _query_embed_cache_key(self, query_text: str) -> str:
        normalized = " ".join(query_text.strip().lower().split())
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
        return f"{self.embed_model_name}::{digest}"

    def _load_query_embed_cache(self) -> None:
        if not self.query_embed_cache_path.is_file():
            return
        try:
            data = json.loads(self.query_embed_cache_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return
            parsed: dict[str, list[float]] = {}
            for key, vec in data.items():
                if not key or not isinstance(vec, list) or not vec:
                    continue
                try:
                    parsed[str(key)] = [float(x) for x in vec]
                except (TypeError, ValueError):
                    continue
            self._query_embed_cache = parsed
            if self.log_timing:
                print(
                    f"[query-embed-cache] loaded {len(self._query_embed_cache)} entries "
                    f"from {self.query_embed_cache_path}"
                )
        except Exception as exc:
            print(
                f"[query-embed-cache] failed to read cache file "
                f"{self.query_embed_cache_path}: {exc}"
            )

    def _prune_query_embed_cache(self) -> None:
        overflow = len(self._query_embed_cache) - self.query_embed_cache_max_entries
        if overflow <= 0:
            return
        for key in list(self._query_embed_cache.keys())[:overflow]:
            self._query_embed_cache.pop(key, None)

    def _save_query_embed_cache(self) -> None:
        if not self.enable_query_embed_cache or not self._query_embed_cache_dirty:
            return
        self._prune_query_embed_cache()
        self.query_embed_cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.query_embed_cache_path.with_suffix(
            self.query_embed_cache_path.suffix + ".tmp"
        )
        payload = json.dumps(self._query_embed_cache, ensure_ascii=False)
        tmp_path.write_text(payload, encoding="utf-8")
        tmp_path.replace(self.query_embed_cache_path)
        self._query_embed_cache_dirty = False
        if self.log_timing:
            print(
                f"[query-embed-cache] saved {len(self._query_embed_cache)} entries "
                f"to {self.query_embed_cache_path}"
            )

    def _resolve_chunks_from_s3(self, keys: list[str]) -> dict[str, str]:
        if not keys:
            return {}

        batch_size = int(os.environ.get("RAG_GET_VECTORS_BATCH", "50"))
        resolved: dict[str, str] = {}
        unique_keys = list(dict.fromkeys(k for k in keys if k))
        unresolved: list[str] = []

        if self.enable_chunk_cache:
            for key in unique_keys:
                cached_text = self._chunk_cache.get(key)
                if cached_text:
                    resolved[key] = cached_text
                    # Promote recently used entries so pruning drops older keys first.
                    self._chunk_cache.pop(key, None)
                    self._chunk_cache[key] = cached_text
                else:
                    unresolved.append(key)
        else:
            unresolved = unique_keys

        for start in range(0, len(unresolved), batch_size):
            batch_keys = unresolved[start : start + batch_size]
            resp = self._client.get_vectors(
                vectorBucketName=self.vector_bucket_name,
                indexName=self.index_name,
                keys=batch_keys,
                returnData=False,
                returnMetadata=True,
            )
            for vec in resp.get("vectors", []) or []:
                key = str(vec.get("key", ""))
                text = _text_from_vector_metadata(vec.get("metadata") or {})
                if key and text:
                    resolved[key] = text
                    if self.enable_chunk_cache and self._chunk_cache.get(key) != text:
                        self._chunk_cache[key] = text
                        self._chunk_cache_dirty = True

        if self.enable_chunk_cache and self._chunk_cache_dirty:
            self._save_chunk_cache()

        if self.log_timing:
            cache_hits = len(unique_keys) - len(unresolved)
            print(
                "[chunk-cache] "
                f"hits={cache_hits} misses={len(unresolved)} resolved={len(resolved)}"
            )

        return resolved

    def _encode_query(self, query: str) -> list[float]:
        from sentence_transformers import SentenceTransformer
        import torch

        if self._embedder is None:
            device = os.environ.get("RAG_EMBED_DEVICE", "cpu")
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
            self._embedder = SentenceTransformer(
                self.embed_model_name, trust_remote_code=True, device=device
            )
            self._embedder.max_seq_length = int(
                os.environ.get("MAX_SEQ_LENGTH", "512" if device == "cpu" else "1024")
            )

        text = query if query.lower().startswith("query:") else f"query: {query}"

        if self.enable_query_embed_cache:
            cache_key = self._query_embed_cache_key(text)
            cached = self._query_embed_cache.get(cache_key)
            if cached:
                self._query_embed_cache.pop(cache_key, None)
                self._query_embed_cache[cache_key] = cached
                if self.log_timing:
                    print("[query-embed-cache] hit")
                return list(cached)

        vec = self._embedder.encode(text, convert_to_numpy=True, show_progress_bar=False)
        out = vec.astype("float32").tolist()

        if self.enable_query_embed_cache:
            self._query_embed_cache[cache_key] = out
            self._query_embed_cache_dirty = True
            self._save_query_embed_cache()
            if self.log_timing:
                print("[query-embed-cache] miss; stored new embedding")

        return out

    def _query_one_stratum(self, spec: dict[str, Any], qvec: list[float]) -> tuple[str, list[RetrievedChunk]]:
        name = str(spec["name"])
        source_file = str(spec["source_file"])
        top_k = int(spec.get("top_k", 1))
        flt = {"source_file": {"$eq": source_file}}

        try:
            resp = self._client.query_vectors(
                vectorBucketName=self.vector_bucket_name,
                indexName=self.index_name,
                topK=top_k,
                queryVector={"float32": qvec},
                returnMetadata=True,
                returnDistance=True,
                filter=flt,
            )
        except Exception as exc:
            print(f"[{name}] query_vectors failed ({source_file!r}): {exc}")
            return name, []

        vecs = resp.get("vectors") or []
        chunks = self._vectors_response_to_chunks(vecs, stratum=name)
        if not chunks:
            print(
                f"[{name}] warning: 0 hits for source_file={source_file!r} "
                "- check SOURCE_FILE_* overrides or corpus coverage."
            )
        print(f"[{name}] retrieved {len(chunks)} chunk(s) (requested top_k={top_k}).")
        return name, chunks

    def _vectors_response_to_chunks(self, items: list[Any], *, stratum: str) -> list[RetrievedChunk]:
        out: list[RetrievedChunk] = []
        for item in items:
            key = str(item.get("key", ""))
            distance = item.get("distance")
            if distance is not None:
                try:
                    distance = float(distance)
                except (TypeError, ValueError):
                    distance = None

            meta = item.get("metadata") or {}
            source_file = str(meta.get("source_file", ""))
            try:
                source_line = int(meta.get("source_line", -1))
            except (TypeError, ValueError):
                source_line = -1

            text = str(
                meta.get("chunk_text")
                or meta.get("text")
                or meta.get("content")
                or ""
            ).strip()

            out.append(
                RetrievedChunk(
                    key=key,
                    stratum=stratum,
                    distance=distance,
                    rerank_score=None,
                    source_file=source_file,
                    source_line=source_line,
                    text=text,
                )
            )
        return out

    def _resolve_text_from_local_kb(self, chunk: RetrievedChunk) -> str:
        paths = _guess_kb_paths(self.kb_dir, chunk.source_file) if chunk.source_file else []
        for path in paths:
            record = _read_jsonl_line(path, chunk.source_line)
            if record:
                text = _chunk_text_from_record(record)
                if text:
                    return text
        if chunk.text:
            return chunk.text
        if chunk.key:
            return (
                f"(no local text resolved for key={chunk.key}; "
                f"source_file={chunk.source_file} line={chunk.source_line})"
            )
        return ""

    def _fill_chunk_text(self, chunks: list[RetrievedChunk]) -> None:
        missing_keys = [chunk.key for chunk in chunks if chunk.key and not (chunk.text or "").strip()]
        if missing_keys:
            s3_text_map = self._resolve_chunks_from_s3(missing_keys)
            for chunk in chunks:
                if chunk.key and not (chunk.text or "").strip():
                    chunk.text = s3_text_map.get(chunk.key, chunk.text)

        for chunk in chunks:
            if not (chunk.text or "").strip():
                chunk.text = self._resolve_text_from_local_kb(chunk)

        for chunk in chunks:
            if not (chunk.text or "").strip():
                chunk.text = (
                    f"(no chunk text available for key={chunk.key}; "
                    f"source_file={chunk.source_file} line={chunk.source_line})"
                )

    def retrieve_weighted_stratified(self, query: str) -> list[RetrievedChunk]:
        qvec = self._encode_query(query)
        print(f"[retrieval] querying {len(self.strata_specs)} strata with merge order: {self.merge_order}")
        with ThreadPoolExecutor(max_workers=max(1, len(self.strata_specs))) as executor:
            stratum_results = list(
                executor.map(lambda spec: self._query_one_stratum(spec, qvec), self.strata_specs)
            )

        by_name = dict(stratum_results)
        merged: list[RetrievedChunk] = []
        seen_keys: set[str] = set()

        for name in self.merge_order:
            for chunk in by_name.get(name, []):
                if chunk.key and chunk.key in seen_keys:
                    continue
                if chunk.key:
                    seen_keys.add(chunk.key)
                merged.append(chunk)

        self._fill_chunk_text(merged)

        return merged

    def _load_reranker(self):
        from sentence_transformers import CrossEncoder
        import torch

        if self._reranker is None:
            rerank_device = os.environ.get("RAG_RERANK_DEVICE", "cpu")
            if rerank_device == "cuda" and not torch.cuda.is_available():
                rerank_device = "cpu"
            rerank_max_length = int(os.environ.get("RERANK_MAX_LENGTH", "256"))
            self._reranker = CrossEncoder(
                self.rerank_model_name,
                device=rerank_device,
                max_length=rerank_max_length,
            )
        return self._reranker

    def _load_answer_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        if self._answer_model is None or self._answer_tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                self.answer_model_path,
                trust_remote_code=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            use_cuda = torch.cuda.is_available()
            if use_cuda:
                model_kwargs: dict[str, Any] = {
                    "trust_remote_code": True,
                    "device_map": "auto",
                    "low_cpu_mem_usage": True,
                    "dtype": torch.bfloat16,
                }
            else:
                # On CPU/Windows, avoid auto device placement that can route shards to
                # an unsupported "disk" target and fail in safetensors.
                model_kwargs = {
                    "trust_remote_code": True,
                    "device_map": None,
                    "low_cpu_mem_usage": False,
                    "dtype": torch.float32,
                }

            try:
                model = AutoModelForCausalLM.from_pretrained(
                    self.answer_model_path, **model_kwargs
                )
            except Exception as exc:
                msg = str(exc).lower()
                if "device disk is invalid" not in msg:
                    raise
                print(
                    "[answer-model] Detected invalid disk offload device; "
                    "retrying with CPU-safe loading settings."
                )
                model = AutoModelForCausalLM.from_pretrained(
                    self.answer_model_path,
                    trust_remote_code=True,
                    device_map=None,
                    low_cpu_mem_usage=False,
                    dtype=torch.float32,
                )

            model.eval()
            self._answer_model = model
            self._answer_tokenizer = tokenizer

        return self._answer_model, self._answer_tokenizer

    def _answer_query(self, query: str, context: str) -> str:
        print("[answer-query] Generating answer from model.")
        model, tokenizer = self._load_answer_model()
        import torch

        user_prompt = (
            f"Retrieved context:\n{context}\n\n"
            f"User question:\n{query}\n\n"
            "Answer the user's question directly."
        )
        messages = [
            {"role": "system", "content": SYSTEM_ANSWER_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        generation_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": self.answer_max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "use_cache": True,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if tokenizer.eos_token_id is not None:
            generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
        if self.answer_temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = self.answer_temperature
            generation_kwargs["top_p"] = self.answer_top_p

        with torch.inference_mode():
            output_ids = model.generate(**generation_kwargs)

        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        print(f"[answer-query] Generated {len(new_tokens)} new tokens.")
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []

        print(f"[rerank] reranking top {self.rerank_top_n or len(chunks)} of {len(chunks)} chunks.")
        rerank_limit = self.rerank_top_n or len(chunks)
        candidate_chunks = chunks[:rerank_limit]
        reranker = self._load_reranker()
        pairs = [(query, chunk.text) for chunk in candidate_chunks]
        rerank_batch_size = int(os.environ.get("RERANK_BATCH_SIZE", "16"))
        scores = reranker.predict(pairs, batch_size=rerank_batch_size, show_progress_bar=False)

        scored_chunks: list[RetrievedChunk] = []
        for chunk, score in zip(candidate_chunks, scores):
            try:
                rerank_score = float(score)
            except (TypeError, ValueError):
                rerank_score = 0.0
            scored_chunks.append(
                RetrievedChunk(
                    key=chunk.key,
                    stratum=chunk.stratum,
                    distance=chunk.distance,
                    rerank_score=rerank_score,
                    source_file=chunk.source_file,
                    source_line=chunk.source_line,
                    text=chunk.text,
                )
            )

        scored_chunks.sort(
            key=lambda chunk: (
                -(chunk.rerank_score if chunk.rerank_score is not None else float("-inf")),
                chunk.distance if chunk.distance is not None else float("inf"),
                chunk.source_file,
                chunk.source_line,
            )
        )
        return scored_chunks

    def run(self, query: str) -> RerankedResult:
        t0 = time.perf_counter()
        retrieved = self.retrieve_weighted_stratified(query)
        t1 = time.perf_counter()
        print("Time to retrieve: {:.2f} seconds".format(t1 - t0))
        if self.enable_rerank:
            reranked = self.rerank(query, retrieved)
            t2 = time.perf_counter()
            print("Time to rerank: {:.2f} seconds".format(t2 - t1))
        else:
            reranked = retrieved[: self.rerank_top_n] if self.rerank_top_n else retrieved
            t2 = time.perf_counter()
            print("[rerank] disabled; using retrieval order.")
            print("Time to rerank: {:.2f} seconds".format(t2 - t1))
        print("[run] retrieved {} chunks, reranked top {} chunks.".format(len(retrieved), len(reranked)))

        context_chunks = reranked
        if self.answer_context_top_n > 0:
            context_chunks = reranked[: self.answer_context_top_n]

        context_max_chars = self.max_context_chars
        if self.answer_context_max_chars > 0:
            context_max_chars = min(context_max_chars, self.answer_context_max_chars)

        context = format_context(context_chunks, max_chars=context_max_chars)
        t_context = time.perf_counter()
        print("Time to format context: {:.2f} seconds".format(t_context - t2))
        answer = self._answer_query(query, context)
        t_answer = time.perf_counter()
        print("Time to generate answer: {:.2f} seconds".format(t_answer - t_context))

        # if self.log_timing:
        t3 = t_answer
        retrieval_ms = (t1 - t0) * 1000.0
        rerank_ms = (t2 - t1) * 1000.0
        answer_ms = (t3 - t2) * 1000.0
        total_ms = (t3 - t0) * 1000.0
        print(
            "[timing] "
            f"retrieval={retrieval_ms:.1f}ms "
            f"rerank={rerank_ms:.1f}ms "
            f"answer={answer_ms:.1f}ms "
            f"total={total_ms:.1f}ms"
        )

        return RerankedResult(query=query, context=context, chunks=reranked, answer=answer)


def build_pipeline_from_env() -> AdvancedRAGPipeline:
    return AdvancedRAGPipeline(
        vector_bucket_name=os.environ.get("RAG_VECTOR_BUCKET", "is469-genai-grp-project"),
        index_name=os.environ.get("RAG_VECTOR_INDEX", "rag-vector-2"),
        region_name=os.environ.get(
            "VECTORS_AWS_DEFAULT_REGION", os.environ.get("AWS_DEFAULT_REGION", "ap-southeast-1")
        ),
        kb_dir=os.environ.get("RAG_KB_DIR", str(_repo_root() / "kb")),
        embed_model_name=os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-small"),
        rerank_model_name=os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
        max_context_chars=int(os.environ.get("RAG_MAX_CONTEXT_CHARS", "12000")),
        rerank_top_n=(int(os.environ["RERANK_TOP_N"]) if os.environ.get("RERANK_TOP_N") else None),
    )


def run_batch_evaluation(
    *,
    input_jsonl: str,
    output_jsonl: str,
    metrics_json: str,
    max_samples: int = 0,
    answer_model_path: str | None = None,
) -> dict[str, Any]:
    if answer_model_path:
        os.environ["ANSWER_MODEL_PATH"] = answer_model_path

    input_path = _resolve_input_path(input_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    sample_limit = max_samples if max_samples > 0 else None
    rows = _load_jsonl(input_path, max_samples=sample_limit)
    if not rows:
        raise ValueError(f"No rows loaded from {input_path}")

    _ensure_vector_credentials()
    pipeline = build_pipeline_from_env()
    assets = build_eval_assets(rows, os.environ.get("RAG_KB_DIR", str(_repo_root() / "kb")))

    predictions: list[dict[str, Any]] = []
    total_latency_ms = 0.0
    total_coverage_score = 0.0

    for row in rows:
        row_id = str(row.get("id", "")).strip()
        source_en = str(row.get("source_en", "")).strip()
        reference_ja = str(row.get("reference_ja", "") or row.get("target_ja", "")).strip()
        if not source_en:
            continue

        t0 = time.perf_counter()
        result = pipeline.run(source_en)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        total_latency_ms += latency_ms

        retrieved_texts = [chunk.text for chunk in result.chunks]
        retrieval_eval = build_retrieval_eval(
            source_en=source_en,
            retrieved_texts=retrieved_texts,
            assets=assets,
        )
        terminology_eval = build_terminology_eval(
            source_en=source_en,
            prediction_ja=result.answer,
            assets=assets,
        )

        strata_hit = len({chunk.stratum for chunk in result.chunks if chunk.stratum})
        coverage_score = round(strata_hit / max(1, len(DEFAULT_MERGE_ORDER)), 4)
        total_coverage_score += coverage_score

        has_error = bool(reference_ja) and (_normalize_ja(result.answer) != _normalize_ja(reference_ja))
        error_check = {
            "has_error": has_error,
            "severity": "minor" if has_error else "none",
            "categories": [],
            "rationale": "auto-check from prediction/reference mismatch",
        }

        predictions.append(
            {
                "id": row_id,
                "source_en": source_en,
                "reference_ja": reference_ja,
                "prediction_ja": result.answer,
                "variant": "advanced_rag_modal",
                "latency_ms": latency_ms,
                "retrieval_ms": None,
                "coverage_score": coverage_score,
                "retrieval_chunks": [
                    {
                        "key": chunk.key,
                        "stratum": chunk.stratum,
                        "distance": chunk.distance,
                        "rerank_score": chunk.rerank_score,
                        "source_file": chunk.source_file,
                        "source_line": chunk.source_line,
                        "text_preview": (chunk.text[:300] + "...") if len(chunk.text) > 300 else chunk.text,
                    }
                    for chunk in result.chunks
                ],
                "retrieval_eval": retrieval_eval,
                "terminology_eval": terminology_eval,
                "error_check": error_check,
                "gold_error_label": assets.gold_error_by_id.get(_canonicalize_id(row_id)) if row_id else None,
            }
        )

    if not predictions:
        raise ValueError("No evaluable rows found in input dataset.")

    metrics: dict[str, Any] = {
        "variant": "advanced_rag_modal",
        "num_samples": len(predictions),
        "avg_latency_ms": round(total_latency_ms / len(predictions), 1),
        "avg_retrieval_ms": None,
        "avg_coverage_score": round(total_coverage_score / len(predictions), 4),
    }

    metrics.update(compute_translation_metrics(predictions))
    try:
        metrics.update(compute_comet_metrics(predictions))
    except Exception as exc:
        metrics["comet_error"] = str(exc)
    metrics.update(compute_retrieval_metrics(predictions))
    metrics.update(compute_terminology_metrics(predictions))
    metrics.update(compute_error_id_metrics(predictions))

    output_path = _resolve_output_path(output_jsonl)
    metrics_path = _resolve_output_path(metrics_json)
    _save_jsonl(output_path, predictions)
    _save_json(metrics_path, metrics)

    return {
        "status": "completed",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "metrics_path": str(metrics_path),
        **metrics,
    }


if modal is not None:
    modal_app = modal.App(MODAL_APP_NAME)
    modal_image = (
        modal.Image.debian_slim(python_version="3.11")
        .uv_pip_install(
            "torch==2.5.1",
            "transformers>=4.46.0",
            "accelerate>=1.2.0",
            "sentence-transformers>=3.3.0",
            "boto3>=1.37.0",
            "sacrebleu>=2.5.0",
            "unbabel-comet>=2.2.2",
            "sentencepiece>=0.2.0",
            "protobuf>=4.25,<6",
        )
        .add_local_dir(_repo_root() / "src", remote_path="/root/src")
        .add_local_dir(_repo_root() / "kb", remote_path="/root/kb")
    )

    modal_models_volume = modal.Volume.from_name("enja-base-models", create_if_missing=True)
    modal_data_volume = modal.Volume.from_name("enja-data", create_if_missing=True)
    modal_results_volume = modal.Volume.from_name("enja-results", create_if_missing=True)

    @modal_app.function(
        image=modal_image,
        gpu="A100",
        timeout=60 * 60,
        volumes={
            str(MODAL_MODELS_DIR): modal_models_volume,
            str(MODAL_DATA_DIR): modal_data_volume,
            str(MODAL_RESULTS_DIR): modal_results_volume,
        },
        secrets=[
            modal.Secret.from_name("enja-hf"),
            modal.Secret.from_name("enja-s3-vectors"),
        ],
    )
    def modal_run_query(query: str, show_chunks: bool = False, answer_model_path: str = "") -> dict[str, Any]:
        sys.path.append("/root")
        if answer_model_path:
            os.environ["ANSWER_MODEL_PATH"] = answer_model_path
        _ensure_vector_credentials()
        pipeline = build_pipeline_from_env()
        result = pipeline.run(query)
        payload: dict[str, Any] = {
            "query": query,
            "answer": result.answer,
            "chunk_count": len(result.chunks),
            "context_chars": len(result.context),
        }
        if show_chunks:
            payload["chunks"] = [
                {
                    "stratum": chunk.stratum,
                    "rerank_score": chunk.rerank_score,
                    "distance": chunk.distance,
                    "key": chunk.key,
                    "source_file": chunk.source_file,
                    "source_line": chunk.source_line,
                    "text_preview": (chunk.text[:240] + "...") if len(chunk.text) > 240 else chunk.text,
                }
                for chunk in result.chunks
            ]
        return payload

    @modal_app.function(
        image=modal_image,
        gpu="A100",
        timeout=2 * 60 * 60,
        volumes={
            str(MODAL_MODELS_DIR): modal_models_volume,
            str(MODAL_DATA_DIR): modal_data_volume,
            str(MODAL_RESULTS_DIR): modal_results_volume,
        },
        secrets=[
            modal.Secret.from_name("enja-hf"),
            modal.Secret.from_name("enja-s3-vectors"),
        ],
    )
    def modal_run_evaluation(
        input_jsonl: str,
        output_jsonl: str,
        metrics_json: str,
        max_samples: int = 0,
        answer_model_path: str = "",
    ) -> dict[str, Any]:
        sys.path.append("/root")
        return run_batch_evaluation(
            input_jsonl=input_jsonl,
            output_jsonl=output_jsonl,
            metrics_json=metrics_json,
            max_samples=max_samples,
            answer_model_path=answer_model_path or None,
        )

    @modal_app.local_entrypoint()
    def modal_entrypoint(
        query: str = "",
        eval_input: str = "",
        output_jsonl: str = "/results/metrics/advanced_rag_pipeline_outputs.modal.jsonl",
        metrics_json: str = "/results/metrics/advanced_rag_pipeline_metrics.modal.json",
        max_samples: int = 0,
        answer_model_path: str = "",
        show_chunks: bool = False,
    ) -> None:
        if eval_input:
            print(
                modal_run_evaluation.remote(
                    input_jsonl=eval_input,
                    output_jsonl=output_jsonl,
                    metrics_json=metrics_json,
                    max_samples=max(0, int(max_samples)),
                    answer_model_path=answer_model_path,
                )
            )
            return

        if not query:
            raise SystemExit("Provide query=<text> or eval_input=<path>.")

        print(
            modal_run_query.remote(
                query=query,
                show_chunks=show_chunks,
                answer_model_path=answer_model_path,
            )
        )
else:
    modal_app = None





def _ensure_vector_credentials() -> None:
    if os.environ.get("VECTORS_AWS_ACCESS_KEY_ID") and os.environ.get(
        "VECTORS_AWS_SECRET_ACCESS_KEY"
    ):
        return
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        return

    raise SystemExit(
        "Missing S3 Vectors credentials. Set VECTORS_AWS_ACCESS_KEY_ID and "
        "VECTORS_AWS_SECRET_ACCESS_KEY in .env or your shell before running."
    )


def main() -> None:
    boot_t0 = time.perf_counter()
    print("Starting advanced_rag_pipeline...")
    parser = argparse.ArgumentParser(
        description="Advanced RAG pipeline with weighted stratified retrieval and reranking."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="English query to retrieve against the S3 vector bucket. If omitted, the script prompts interactively.",
    )
    parser.add_argument(
        "--use-modal",
        action="store_true",
        help="Run on Modal GPU instead of local execution.",
    )
    parser.add_argument(
        "--modal-eval-input",
        default="",
        help="JSONL input path for Modal batch evaluation. Example: /data/splits/test_v1.jsonl",
    )
    parser.add_argument(
        "--modal-output-jsonl",
        default="/results/metrics/advanced_rag_pipeline_outputs.modal.jsonl",
        help="Output JSONL path when --modal-eval-input is set.",
    )
    parser.add_argument(
        "--modal-metrics-json",
        default="/results/metrics/advanced_rag_pipeline_metrics.modal.json",
        help="Metrics JSON path when --modal-eval-input is set.",
    )
    parser.add_argument(
        "--modal-max-samples",
        type=int,
        default=0,
        help="Optional cap for Modal batch evaluation rows.",
    )
    parser.add_argument(
        "--modal-answer-model-path",
        default="",
        help="Optional answer model path/id override for Modal execution.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Keep the process alive and answer multiple questions in one session (type 'exit' to quit).",
    )
    parser.add_argument("--show-chunks", action="store_true", help="Print reranked chunks and scores.")
    args = parser.parse_args()

    query = args.query
    interactive_mode = bool(args.interactive or not query)

    if args.use_modal:
        if modal is None or modal_app is None:
            raise SystemExit(
                "Modal support is unavailable. Install 'modal' and retry with 'modal setup' completed."
            )

        if args.modal_eval_input:
            payload = modal_run_evaluation.remote(
                input_jsonl=args.modal_eval_input,
                output_jsonl=args.modal_output_jsonl,
                metrics_json=args.modal_metrics_json,
                max_samples=max(0, int(args.modal_max_samples)),
                answer_model_path=args.modal_answer_model_path,
            )
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return

        if not query:
            if not sys.stdin.isatty():
                raise SystemExit(
                    "No query provided. Pass a query argument or run interactively from a terminal."
                )
            query = input("Enter a question for Japanese translation: ").strip()
        if not query:
            raise SystemExit("No query provided.")

        payload = modal_run_query.remote(
            query=query,
            show_chunks=bool(args.show_chunks),
            answer_model_path=args.modal_answer_model_path,
        )
        print("\n=== Answer (Modal) ===\n")
        print(str(payload.get("answer", "")))
        if args.show_chunks:
            print("\n=== Reranked chunks (Modal) ===\n")
            for chunk in payload.get("chunks", []):
                print(json.dumps(chunk, ensure_ascii=False))
        return

    if interactive_mode and not sys.stdin.isatty() and not query:
        raise SystemExit(
            "No query provided. Run this script in an interactive terminal or pass the query as an argument."
        )
    if not interactive_mode and not query:
        raise SystemExit("No query provided.")

    _ensure_vector_credentials()

    build_t0 = time.perf_counter()
    pipeline = build_pipeline_from_env()
    print("Time to build pipeline: {:.2f} seconds".format(time.perf_counter() - build_t0))

    if _env_enabled("RAG_PRELOAD_MODELS", default=interactive_mode):
        print("[warmup] preloading embedder/reranker/answer model (set RAG_PRELOAD_MODELS=false to disable)")
        pipeline.warmup_models()

    print("Time to first prompt-ready state: {:.2f} seconds".format(time.perf_counter() - boot_t0))
    next_query = query
    while True:
        if not next_query:
            next_query = input("Enter a question for Japanese translation (or 'exit' to quit): ").strip()

        if not next_query:
            if interactive_mode:
                print("Exiting interactive mode.")
                break
            raise SystemExit("No query provided.")

        if next_query.lower() in {"exit", "quit", "q"}:
            print("Exiting interactive mode.")
            break

        result = pipeline.run(next_query)

        print("\n=== Answer ===\n")
        print(result.answer)

        if args.show_chunks:
            print("\n=== Context ===\n")
            print(result.context)

        if args.show_chunks:
            print("\n=== Reranked chunks ===\n")
            for i, chunk in enumerate(result.chunks, start=1):
                print(
                    json.dumps(
                        {
                            "rank": i,
                            "stratum": chunk.stratum,
                            "rerank_score": chunk.rerank_score,
                            "distance": chunk.distance,
                            "key": chunk.key,
                            "source_file": chunk.source_file,
                            "source_line": chunk.source_line,
                            "text_preview": (chunk.text[:240] + "...") if len(chunk.text) > 240 else chunk.text,
                        },
                        ensure_ascii=False,
                    )
                )

        if not interactive_mode:
            break
        next_query = None


if __name__ == "__main__":
    main()