#!/usr/bin/env python3
"""
Agentic RAG translation pipeline using OpenRouter API.

No GPU required — LLM calls go through OpenRouter, retrieval through S3 Vectors.

Usage:
    python scripts/run_agentic.py --limit 250 --workers 10
    python scripts/run_agentic.py --config configs/s3_inference_agentic.yaml
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_print_lock = Lock()


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def _load_yaml(path: str) -> dict:
    import yaml

    p = Path(path)
    if not p.exists():
        p = ROOT / path
    if not p.exists():
        p = ROOT / "configs" / Path(path).name
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _process_row(
    idx: int,
    total: int,
    row: dict,
    client,
    glossary_path: Path,
    retriever,
    agent_cfg: dict,
    models_cfg: dict,
    eval_assets,
    kb_dir: str,
) -> dict:
    from src.agents.tools import ToolExecutor
    from src.agents.agentic_rag_v2 import translate_with_agent, detect_error_with_api
    from src.eval.s3_eval import (
        build_retrieval_eval,
        build_terminology_eval,
        canonicalize_id,
    )

    row_id = row.get("id", str(idx))
    source_en = row.get("source_en", "")
    reference_ja = row.get("target_ja", "")

    executor = ToolExecutor(glossary_path=glossary_path, retriever=retriever)

    t0 = time.time()
    result = translate_with_agent(
        client=client,
        executor=executor,
        source_en=source_en,
        agent_cfg=agent_cfg,
        models_cfg=models_cfg,
    )

    error_check = detect_error_with_api(
        client=client,
        source_en=source_en,
        candidate_ja=result.translation,
        model=models_cfg.get("critic", client.default_model),
    )
    latency = time.time() - t0

    retrieved_texts = [
        tc.get("result_preview", "")
        for step in result.trace
        for tc in step.tool_calls
        if tc.get("name") == "search_knowledge_base"
    ]
    glossary_texts = [
        tc.get("result_preview", "")
        for step in result.trace
        for tc in step.tool_calls
        if tc.get("name") == "lookup_glossary"
    ]
    all_retrieved = retrieved_texts + glossary_texts

    retrieval_eval = build_retrieval_eval(
        source_en=source_en,
        retrieved_texts=all_retrieved,
        assets=eval_assets,
    )

    terminology_eval = build_terminology_eval(
        source_en=source_en,
        prediction_ja=result.translation,
        assets=eval_assets,
    )

    gold_error_label = eval_assets.gold_error_by_id.get(
        canonicalize_id(row_id)
    )

    rewrite_steps = sum(
        1 for t in result.trace if t.step_type == "rewrite"
    )
    revision_steps = sum(
        1 for t in result.trace if t.step_type == "revision"
    )

    with _print_lock:
        print(
            f"[{idx + 1}/{total}] id={row_id}  "
            f"coverage={result.coverage_score:.2f}  "
            f"tools={result.total_tool_calls}  "
            f"latency={latency:.1f}s"
        )

    return {
        "id": row_id,
        "source_en": source_en,
        "prediction_ja": result.translation,
        "reference_ja": reference_ja,
        "variant": "s3_agentic",
        "latency_ms": round(latency * 1000, 1),
        "coverage_score": result.coverage_score,
        "total_tool_calls": result.total_tool_calls,
        "rewrite_steps": rewrite_steps,
        "revision_steps": revision_steps,
        "error_check": {
            "has_error": error_check.has_error,
            "severity": error_check.severity,
            "categories": error_check.categories,
            "rationale": error_check.rationale,
        },
        "gold_error_label": gold_error_label,
        "retrieval_eval": retrieval_eval,
        "terminology_eval": terminology_eval,
        "agent_trace": [
            {
                "step_type": t.step_type,
                "candidate_ja": t.candidate_ja,
                "critic_coverage_score": t.critic_coverage_score,
                "critic_has_error": t.critic_has_error,
                "critic_feedback": t.critic_feedback,
                "tool_calls": t.tool_calls,
            }
            for t in result.trace
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic RAG translation")
    parser.add_argument(
        "--config", default="configs/s3_inference_agentic.yaml"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Process only first N rows"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Parallel workers (default 10)",
    )
    parser.add_argument(
        "--skip-comet",
        action="store_true",
        help="Skip COMET (requires GPU + large model download)",
    )
    args = parser.parse_args()

    _load_dotenv()

    from src.agents.openrouter_client import OpenRouterClient
    from src.eval.s3_eval import (
        build_eval_assets,
        compute_error_id_metrics,
        compute_retrieval_metrics,
        compute_terminology_metrics,
    )

    cfg = _load_yaml(args.config)
    models_cfg = cfg.get("models", {})
    io_cfg = cfg.get("io", {})
    agent_cfg = cfg.get("agent", {})
    retrieval_cfg = cfg.get("retrieval", {})

    print(
        f"Models: translator={models_cfg.get('translator')} "
        f"critic={models_cfg.get('critic')}"
    )
    print(f"Workers: {args.workers}")

    client = OpenRouterClient(
        default_model=models_cfg.get(
            "translator", "anthropic/claude-sonnet-4-6"
        )
    )

    retriever = None
    if retrieval_cfg.get("enabled", False):
        try:
            from src.retrieval.s3_vectors_rag import S3VectorsRAGRetriever

            kb_dir = retrieval_cfg.get("kb_dir", "kb")
            if not Path(kb_dir).is_absolute():
                kb_dir = str(ROOT / kb_dir)

            retriever = S3VectorsRAGRetriever(
                vector_bucket_name=retrieval_cfg["vector_bucket_name"],
                index_name=retrieval_cfg["index_name"],
                region_name=retrieval_cfg.get("region", "ap-southeast-1"),
                kb_dir=kb_dir,
                embed_model_name=retrieval_cfg.get("embed_model"),
                top_k=int(retrieval_cfg.get("top_k", 10)),
                max_context_chars=int(
                    retrieval_cfg.get("max_context_chars", 12000)
                ),
            )
            print(
                f"S3 Vectors retriever: "
                f"{retrieval_cfg['vector_bucket_name']}/{retrieval_cfg['index_name']}"
            )
        except Exception as e:
            print(f"WARNING: Could not init S3 Vectors retriever: {e}")
            print("Continuing with glossary-only tools.")

    glossary_path = ROOT / "kb" / "glossary.csv"
    from src.agents.tools import ToolExecutor
    tmp_executor = ToolExecutor(glossary_path=glossary_path, retriever=retriever)
    print(f"Glossary: {len(tmp_executor.glossary)} terms")
    print(f"Available tools: {tmp_executor.available_tool_names}")
    del tmp_executor

    input_path = io_cfg.get("input_path", "data/splits/test_v1.jsonl")
    p = Path(input_path)
    if not p.exists():
        p = ROOT / input_path
    rows = _load_jsonl(str(p))
    if args.limit:
        rows = rows[: args.limit]
    print(f"Loaded {len(rows)} rows from {p}")

    kb_dir = retrieval_cfg.get("kb_dir", "kb")
    if not Path(kb_dir).is_absolute():
        kb_dir = str(ROOT / kb_dir)
    eval_assets = build_eval_assets(rows, kb_dir)
    print(
        f"Eval assets: {len(eval_assets.glossary_entries)} glossary, "
        f"{len(eval_assets.gold_error_by_id)} gold error labels"
    )

    est_per_row = 55 / max(args.workers, 1)
    est_total = est_per_row * len(rows)
    print(f"Estimated time: ~{est_total / 60:.0f} min ({args.workers} workers)\n")

    t_start = time.time()

    if args.workers <= 1:
        predictions = []
        for i, row in enumerate(rows):
            pred = _process_row(
                i, len(rows), row, client, glossary_path,
                retriever, agent_cfg, models_cfg, eval_assets, kb_dir,
            )
            predictions.append(pred)
    else:
        predictions_map: dict[int, dict] = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    _process_row,
                    i, len(rows), row, client, glossary_path,
                    retriever, agent_cfg, models_cfg, eval_assets, kb_dir,
                ): i
                for i, row in enumerate(rows)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    predictions_map[idx] = future.result()
                except Exception as exc:
                    row_id = rows[idx].get("id", str(idx))
                    print(f"ERROR on row {idx} (id={row_id}): {exc}")
                    predictions_map[idx] = {
                        "id": row_id,
                        "source_en": rows[idx].get("source_en", ""),
                        "prediction_ja": "",
                        "reference_ja": rows[idx].get("target_ja", ""),
                        "variant": "s3_agentic",
                        "error": str(exc),
                    }
        predictions = [predictions_map[i] for i in range(len(rows))]

    wall_time = time.time() - t_start

    output_path = Path(
        io_cfg.get("output_path", "results/metrics/s3_agentic_outputs.jsonl")
    )
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    valid = [p for p in predictions if p.get("prediction_ja")]
    n = max(len(valid), 1)

    avg_coverage = sum(p.get("coverage_score", 0) for p in valid) / n
    avg_latency_ms = sum(p.get("latency_ms", 0) for p in valid) / n
    total_tools = sum(p.get("total_tool_calls", 0) for p in valid)
    total_rewrites = sum(p.get("rewrite_steps", 0) for p in valid)
    total_revisions = sum(p.get("revision_steps", 0) for p in valid)

    summary: dict = {
        "variant": "s3_agentic",
        "num_samples": len(valid),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "avg_retrieval_ms": "—",
        "avg_coverage_score": round(avg_coverage, 4),
        "total_rewrite_steps": total_rewrites,
        "total_revision_steps": total_revisions,
    }

    if any(p.get("reference_ja") for p in valid):
        try:
            from sacrebleu.metrics import BLEU, CHRF

            refs = [[p["reference_ja"] for p in valid]]
            hyps = [p["prediction_ja"] for p in valid]
            try:
                bleu_result = BLEU(tokenize="ja-mecab").corpus_score(hyps, refs)
                summary["bleu_tokenizer"] = "ja-mecab"
            except Exception:
                bleu_result = BLEU(tokenize="char").corpus_score(hyps, refs)
                summary["bleu_tokenizer"] = "char"
            chrfpp_result = CHRF(word_order=2).corpus_score(hyps, refs)
            summary["bleu"] = round(bleu_result.score, 2)
            summary["chrfpp"] = round(chrfpp_result.score, 2)
        except ImportError:
            pass

    if not args.skip_comet and any(p.get("reference_ja") for p in valid):
        try:
            from src.eval.s3_eval import compute_comet_metrics
            print("\nComputing COMET (this may take a while)...")
            comet_metrics = compute_comet_metrics(valid)
            summary.update(comet_metrics)
        except Exception as exc:
            summary["comet_error"] = str(exc)
            print(f"COMET skipped: {exc}")

    term_metrics = compute_terminology_metrics(valid)
    summary.update(term_metrics)

    retrieval_metrics = compute_retrieval_metrics(valid)
    summary.update(retrieval_metrics)

    error_metrics = compute_error_id_metrics(valid)
    summary.update(error_metrics)

    summary["wall_time_s"] = round(wall_time, 1)
    summary["wall_time_min"] = round(wall_time / 60, 1)
    summary["workers"] = args.workers
    summary["total_tool_calls"] = total_tools
    summary["avg_tool_calls"] = round(total_tools / n, 1)

    metrics_path = output_path.with_name(
        output_path.stem.replace("_outputs", "_metrics") + ".json"
    )
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"EVAL COMPLETE — {len(valid)} translations in {wall_time:.0f}s")
    print(f"{'=' * 60}")

    table_keys = [
        ("Num Samples", "num_samples"),
        ("Avg Latency (ms)", "avg_latency_ms"),
        ("Avg Retrieval (ms)", "avg_retrieval_ms"),
        ("Avg Coverage Score", "avg_coverage_score"),
        ("Total Rewrite Steps", "total_rewrite_steps"),
        ("Total Revision Steps", "total_revision_steps"),
        ("BLEU (char)", "bleu"),
        ("ChrF++", "chrfpp"),
        ("COMET", "comet"),
        ("Terminology Samples", "terminology_eval_samples"),
        ("Terminology Terms Total", "terminology_term_total"),
        ("Terminology Correct Terms", "terminology_correct_terms"),
        ("Terminology Accuracy", "terminology_accuracy"),
        ("Retrieval Samples", "retrieval_eval_samples"),
        ("Retrieval Hit@K", "retrieval_hit_at_k"),
        ("Retrieval Recall@K", "retrieval_recall_at_k"),
        ("Error Eval Samples", "error_id_eval_samples"),
        ("Error Binary F1", "error_binary_f1"),
        ("Error Category Macro F1", "error_category_macro_f1"),
    ]

    print(f"\n{'Metric':<30} {'S3 Agentic':>15}")
    print("-" * 47)
    for label, key in table_keys:
        val = summary.get(key, "—")
        if val is None:
            val = "—"
        print(f"{label:<30} {str(val):>15}")

    print(f"\nOutputs:  {output_path}")
    print(f"Metrics:  {metrics_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
