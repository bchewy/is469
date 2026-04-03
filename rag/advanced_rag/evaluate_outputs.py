from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from src.eval.s3_eval import (
    build_eval_assets,
    build_retrieval_eval,
    build_terminology_eval,
    compute_comet_metrics,
    compute_error_id_metrics,
    compute_retrieval_metrics,
    compute_terminology_metrics,
)


ERROR_LABEL_PREFIXES = ("annot-", "engjap-", "tm-")


def _repo_root() -> Path:
    for path in [Path.cwd(), *Path.cwd().parents]:
        if (path / "src").is_dir():
            return path
    return Path.cwd()


# Ensure repo root is in sys.path for imports to work from any directory
_root = _repo_root()
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


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


def _save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
    has_error = not bool(str(prediction_ja).strip())
    return {
        "has_error": has_error,
        "categories": [],
        "severity": "minor" if has_error else "none",
    }


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
    return metrics


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

        chunks = [_chunk_to_eval_dict(chunk) for chunk in result.chunks]
        row: dict[str, Any] = {
            "id": sample["id"],
            "source_en": source_en,
            "reference_ja": sample["reference_ja"],
            "prediction_ja": result.answer,
            "retrieval_chunks": chunks,
            "latency_ms": latency_ms,
            "error_check": _default_error_check(result.answer),
        }
        rows.append(row)
        print(f"[{i}/{len(eval_set)}] Evaluated query id={sample['id']}")

    if save_to:
        _save_jsonl(save_to, rows)
        print(f"Saved generated outputs: {save_to}")

    return rows, save_to


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
        default=str(_repo_root() / "kb" / "eng-jap.tsv"),
        help="TSV dataset path used in --run-pipeline mode (expects id, source_en, *, reference_ja columns).",
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
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    output_env = os.environ.get("EVAL_OUTPUT")
    run_pipeline_mode = bool(args.run_pipeline or (args.output is None and not output_env))

    output_path: Path | None = None
    if run_pipeline_mode:
        save_path = Path(args.save_generated) if args.save_generated.strip() else None
        rows, output_path = generate_pipeline_outputs(
            dataset_path=Path(args.dataset),
            kb_dir=args.kb_dir,
            max_samples=args.max_samples,
            save_to=save_path,
        )
        reranked_metrics = evaluate_rows(rows, kb_dir=args.kb_dir, output_path=output_path)
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
    print("")
    print("No baseline was provided, so this run can be scored but not compared.")


if __name__ == "__main__":
    main()
