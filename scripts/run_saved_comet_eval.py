#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.eval.s3_eval import compute_comet_metrics


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _score(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    print(f"[comet] start {name}: rows={len(rows)}", flush=True)
    metrics = compute_comet_metrics(rows)
    print(f"[comet] done  {name}: comet={metrics.get('comet')}", flush=True)
    return metrics


def _score_s2(root: Path, summary: dict[str, Any], train_samples: int, test_samples: int) -> None:
    metrics_path = root / "results/metrics/reranker.modal.train250.test2000.metrics.json"
    outputs_path = root / "results/metrics/advanced_rag_pipeline_outputs.modal.2250.jsonl"
    if not metrics_path.exists() or not outputs_path.exists():
        print("[comet] skip s2: missing metrics or outputs file", flush=True)
        return

    rows = _load_jsonl(outputs_path)
    if len(rows) < train_samples:
        raise ValueError(
            f"S2 outputs file has {len(rows)} rows, expected at least {train_samples} "
            f"for the canonical 250-row S2 split."
        )

    train_rows = rows[:train_samples]
    train_comet = _score("s2_train250", train_rows)

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload.setdefault("train", {})["comet"] = train_comet["comet"]
    payload["train"]["comet_source"] = "wmt22-comet-da"
    if len(rows) >= train_samples + test_samples:
        test_rows = rows[train_samples : train_samples + test_samples]
        test_comet = _score("s2_test2000", test_rows)
        payload.setdefault("test", {})["comet"] = test_comet["comet"]
        payload["test"]["comet_source"] = "wmt22-comet-da"
        summary["s2_test2000"] = test_comet
    else:
        payload.setdefault("test", {})["comet_source"] = "unavailable_saved_rows"
        payload["test"]["comet_rows_available"] = max(len(rows) - train_samples, 0)
    _write_json(metrics_path, payload)

    summary["s2_train250"] = train_comet


def _score_single(root: Path, summary: dict[str, Any], *, name: str, metrics_rel: str, outputs_rel: str) -> None:
    metrics_path = root / metrics_rel
    outputs_path = root / outputs_rel
    if not metrics_path.exists() or not outputs_path.exists():
        print(f"[comet] skip {name}: missing metrics or outputs file", flush=True)
        return

    rows = _load_jsonl(outputs_path)
    comet = _score(name, rows)

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["comet"] = comet["comet"]
    payload["comet_source"] = "wmt22-comet-da"
    _write_json(metrics_path, payload)

    summary[name] = comet


def main() -> None:
    parser = argparse.ArgumentParser(description="Run COMET on saved S2/S3 output files.")
    parser.add_argument("--root", default=".", help="Repo root. Defaults to current directory.")
    parser.add_argument("--skip-s2", action="store_true", help="Skip S2 split scoring.")
    parser.add_argument("--skip-s3-eval250", action="store_true", help="Skip s3_agentic_eval250 scoring.")
    parser.add_argument("--skip-s3-outputs", action="store_true", help="Skip s3_agentic_outputs scoring.")
    parser.add_argument("--skip-s3-erroreval", action="store_true", help="Skip s3_agentic_erroreval scoring.")
    parser.add_argument("--summary-path", default="results/metrics/s2_s3_comet_eval.json")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    summary: dict[str, Any] = {}

    if not args.skip_s2:
        _score_s2(root, summary, train_samples=250, test_samples=2000)
    if not args.skip_s3_eval250:
        _score_single(
            root,
            summary,
            name="s3_eval250",
            metrics_rel="results/metrics/s3_agentic_eval250_metrics.json",
            outputs_rel="results/metrics/s3_agentic_eval250_outputs.jsonl",
        )
    if not args.skip_s3_outputs:
        _score_single(
            root,
            summary,
            name="s3_outputs250",
            metrics_rel="results/metrics/s3_agentic_metrics.json",
            outputs_rel="results/metrics/s3_agentic_outputs.jsonl",
        )
    if not args.skip_s3_erroreval:
        _score_single(
            root,
            summary,
            name="s3_erroreval250",
            metrics_rel="results/metrics/s3_agentic_erroreval_metrics.json",
            outputs_rel="results/metrics/s3_agentic_erroreval_outputs.jsonl",
        )

    summary_path = root / args.summary_path
    _write_json(summary_path, summary)
    print(f"[comet] wrote summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
