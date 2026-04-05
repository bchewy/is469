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
import json
import os
import sys
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.aws_profiles import s3vectors_client


def _repo_root() -> Path:
    for path in [Path.cwd(), *Path.cwd().parents]:
        if (path / "src" / "retrieval").is_dir():
            return path
    return Path.cwd()


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
        self.rerank_top_n = rerank_top_n
        self.answer_model_path = os.environ.get(
            "ANSWER_MODEL_PATH", os.environ.get("ANSWER_BASE_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
        )
        self.answer_max_new_tokens = int(os.environ.get("ANSWER_MAX_NEW_TOKENS", "256"))
        self.answer_temperature = float(os.environ.get("ANSWER_TEMPERATURE", "0.2"))
        self.answer_top_p = float(os.environ.get("ANSWER_TOP_P", "0.9"))

        self._client = s3vectors_client(region_name=region_name)
        self._embedder = None
        self._reranker = None
        self._answer_model = None
        self._answer_tokenizer = None

    def _resolve_chunks_from_s3(self, keys: list[str]) -> dict[str, str]:
        if not keys:
            return {}

        batch_size = int(os.environ.get("RAG_GET_VECTORS_BATCH", "50"))
        resolved: dict[str, str] = {}
        unique_keys = list(dict.fromkeys(k for k in keys if k))

        for start in range(0, len(unique_keys), batch_size):
            batch_keys = unique_keys[start : start + batch_size]
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
        vec = self._embedder.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return vec.astype("float32").tolist()

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

        if self._reranker is None:
            self._reranker = CrossEncoder(self.rerank_model_name)
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

            model_kwargs: dict[str, Any] = {
                "trust_remote_code": True,
                "device_map": "auto",
                "low_cpu_mem_usage": True,
            }
            if torch.cuda.is_available():
                model_kwargs["torch_dtype"] = torch.bfloat16
            else:
                model_kwargs["torch_dtype"] = torch.float32

            model = AutoModelForCausalLM.from_pretrained(self.answer_model_path, **model_kwargs)
            model.eval()
            self._answer_model = model
            self._answer_tokenizer = tokenizer

        return self._answer_model, self._answer_tokenizer

    def _answer_query(self, query: str, context: str) -> str:
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
            "pad_token_id": tokenizer.pad_token_id,
        }
        if self.answer_temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = self.answer_temperature
            generation_kwargs["top_p"] = self.answer_top_p

        with torch.no_grad():
            output_ids = model.generate(**generation_kwargs)

        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []

        rerank_limit = self.rerank_top_n or len(chunks)
        candidate_chunks = chunks[:rerank_limit]
        reranker = self._load_reranker()
        pairs = [(query, chunk.text) for chunk in candidate_chunks]
        scores = reranker.predict(pairs)

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
        retrieved = self.retrieve_weighted_stratified(query)
        reranked = self.rerank(query, retrieved)
        context = format_context(reranked, max_chars=self.max_context_chars)
        answer = self._answer_query(query, context)
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
    parser = argparse.ArgumentParser(
        description="Advanced RAG pipeline with weighted stratified retrieval and reranking."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="English query to retrieve against the S3 vector bucket. If omitted, the script prompts interactively.",
    )
    parser.add_argument("--show-chunks", action="store_true", help="Print reranked chunks and scores.")
    args = parser.parse_args()

    query = args.query
    if not query:
        if not sys.stdin.isatty():
            raise SystemExit(
                "No query provided. Run this script in an interactive terminal or pass the query as an argument."
            )
        query = input("Enter a question for Japanese translation: ").strip()

    if not query:
        raise SystemExit("No query provided.")

    _ensure_vector_credentials()

    pipeline = build_pipeline_from_env()
    result = pipeline.run(query)

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


if __name__ == "__main__":
    main()