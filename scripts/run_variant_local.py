#!/usr/bin/env python3
"""Local S0/S1 inference runner for reproducible GPU reruns."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

SYSTEM_PROMPT = (
    "You are a professional English-to-Japanese translator. "
    "Translate the following English text into natural, fluent Japanese. "
    "Preserve the original meaning and tone."
)


def _resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _load_yaml(path: str) -> dict:
    cfg_path = _resolve_repo_path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _compute_translation_metrics(predictions: list[dict]) -> dict:
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


def _load_glossary(glossary_path: Path) -> list[tuple[str, str]]:
    if not glossary_path.exists():
        print(f"No glossary file found at {glossary_path}")
        return []

    pairs: list[tuple[str, str]] = []
    with glossary_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            en = row.get("source_term_en", "").strip().lower()
            ja = row.get("approved_ja", "").strip()
            if en and ja:
                pairs.append((en, ja))
    print(f"Loaded {len(pairs)} glossary terms from {glossary_path}")
    return pairs


def _copy_glossary_snapshot(glossary_path: Path) -> Path:
    dst = REPO_ROOT / "results" / "metrics" / "glossary_used_for_s0_s1.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(glossary_path, dst)
    return dst


def _qual_entry(pred: dict, score: float) -> dict:
    return {
        "id": pred.get("id", ""),
        "source_en": pred["source_en"],
        "prediction_ja": pred["prediction_ja"],
        "reference_ja": pred.get("reference_ja", ""),
        "comet_score": round(score, 4),
        "glossary_matches": pred.get("glossary_matches", []),
    }


def run(variant: str, config: str) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    cfg = _load_yaml(config)
    model_cfg = cfg.get("model", {})
    io_cfg = cfg.get("io", {})
    gen_cfg = cfg.get("generation", {})

    base_model_id = model_cfg.get("base_model_id", "Qwen/Qwen2.5-0.5B-Instruct")
    local_model_path = model_cfg.get("model_local_path")
    model_path = _resolve_repo_path(local_model_path) if local_model_path else Path(base_model_id)
    if not model_path.exists():
        model_path = Path(base_model_id)

    input_path = _resolve_repo_path(io_cfg.get("input_path", "data/splits/test_v1.jsonl"))
    output_path = _resolve_repo_path(io_cfg.get("output_path", f"results/metrics/{variant}_outputs.jsonl"))
    metrics_path = _resolve_repo_path(
        io_cfg.get("metrics_path", f"results/metrics/{variant}_metrics.json")
    )
    qual_path = _resolve_repo_path(
        io_cfg.get("qualitative_path", f"results/qualitative_examples/{variant}.json")
    )
    glossary_path = _resolve_repo_path(io_cfg.get("glossary_path", "kb/glossary.csv"))

    adapter_dir_str = model_cfg.get("adapter_dir", "models/adapter-translation-qwen25-0p5b/final")
    adapter_dir = _resolve_repo_path(adapter_dir_str)
    print(f"Variant: {variant}")
    print(f"Model: {model_path}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    rows = _load_jsonl(input_path)
    print(f"Loaded {len(rows)} eval rows")

    tokenizer = None
    tokenizer_source = model_path
    tokenizer_candidates = [model_path]
    if variant == "s1" and adapter_dir.exists():
        tokenizer_candidates = [adapter_dir, model_path]

    for candidate in tokenizer_candidates:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(candidate),
                trust_remote_code=True,
                token=os.environ.get("HF_TOKEN"),
            )
            tokenizer_source = candidate
            break
        except Exception as exc:
            print(f"Tokenizer load failed for {candidate}: {exc}")

    if tokenizer is None:
        raise RuntimeError("Unable to load tokenizer from either adapter or base model path")

    print(f"Tokenizer: {tokenizer_source}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
        torch_dtype=torch.bfloat16,
    )

    if variant == "s1" and adapter_dir.exists():
        print(f"Loading LoRA adapter from {adapter_dir}")
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    elif variant == "s1":
        print(f"WARNING: adapter_dir {adapter_dir} not found, running without adapter")

    model.eval()

    seed = cfg.get("project", {}).get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    predictions: list[dict] = []
    total_latency_ms = 0.0
    do_sample = bool(gen_cfg.get("do_sample", False))
    num_beams = int(gen_cfg.get("num_beams", 1))
    max_new_tokens = int(gen_cfg.get("max_new_tokens", 512))
    temperature = float(gen_cfg.get("temperature", 0.1))
    top_p = float(gen_cfg.get("top_p", 0.95))
    infer_batch_size = int(gen_cfg.get("batch_size", 16))

    for batch_start in range(0, len(rows), infer_batch_size):
        batch_rows = rows[batch_start : batch_start + infer_batch_size]
        input_texts = []
        for row in batch_rows:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row.get("source_en", "")},
            ]
            input_texts.append(
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )

        inputs = tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        t0 = time.time()
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_beams": max(1, num_beams),
            "pad_token_id": tokenizer.pad_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
        with torch.no_grad():
            output_ids = model.generate(**generation_kwargs)
        batch_latency_ms = round((time.time() - t0) * 1000, 1)
        per_sample_ms = round(batch_latency_ms / len(batch_rows), 1)
        total_latency_ms += batch_latency_ms

        prompt_len = inputs["input_ids"].shape[1]
        for idx, row in enumerate(batch_rows):
            new_tokens = output_ids[idx][prompt_len:]
            translation = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            predictions.append(
                {
                    "id": row.get("id", ""),
                    "source_en": row.get("source_en", ""),
                    "prediction_ja": translation,
                    "reference_ja": row.get("target_ja", ""),
                    "variant": variant,
                    "latency_ms": per_sample_ms,
                }
            )

        if (batch_start // infer_batch_size) % 10 == 0:
            batch_num = batch_start // infer_batch_size + 1
            total_batches = (len(rows) + infer_batch_size - 1) // infer_batch_size
            print(f"Batch {batch_num}/{total_batches} done")

    metrics: dict = {
        "variant": variant,
        "num_samples": len(predictions),
        "avg_latency_ms": round(total_latency_ms / max(len(predictions), 1), 1),
    }

    has_refs = any(p["reference_ja"] for p in predictions)
    comet_scores = None
    if has_refs:
        metrics.update(_compute_translation_metrics(predictions))
        print(
            f"BLEU ({metrics['bleu_tokenizer']}): {metrics['bleu']}, "
            f"chrF++: {metrics['chrfpp']}"
        )

        try:
            from comet import download_model, load_from_checkpoint

            comet_path = download_model("Unbabel/wmt22-comet-da")
            comet_model = load_from_checkpoint(comet_path)
            comet_data = [
                {
                    "src": p["source_en"],
                    "mt": p["prediction_ja"],
                    "ref": p["reference_ja"],
                }
                for p in predictions
            ]
            comet_output = comet_model.predict(
                comet_data,
                batch_size=32,
                gpus=1 if torch.cuda.is_available() else 0,
            )
            comet_scores = comet_output.scores
            metrics["comet"] = round(float(comet_output.system_score), 4)
            print(f"COMET: {metrics['comet']}")
            for idx, pred in enumerate(predictions):
                pred["comet_score"] = round(float(comet_scores[idx]), 4)
        except Exception as exc:
            print(f"COMET evaluation failed: {exc}")

        glossary = _load_glossary(glossary_path)
        if glossary:
            glossary_snapshot_path = _copy_glossary_snapshot(glossary_path)
            metrics["glossary_entries_used"] = len(glossary)
            metrics["glossary_snapshot_path"] = str(glossary_snapshot_path.relative_to(REPO_ROOT))

            term_hits = 0
            term_correct = 0
            terminology_samples = 0
            for pred in predictions:
                src_lower = pred["source_en"].lower()
                pred_ja = pred["prediction_ja"]
                matched_terms = []
                row_has_term = False
                for en_term, ja_term in glossary:
                    if en_term in src_lower:
                        row_has_term = True
                        term_hits += 1
                        is_correct = ja_term in pred_ja
                        if is_correct:
                            term_correct += 1
                        matched_terms.append((en_term, ja_term, is_correct))
                if row_has_term:
                    terminology_samples += 1
                pred["glossary_matches"] = matched_terms

            metrics["terminology_samples"] = terminology_samples
            if term_hits > 0:
                metrics["term_accuracy"] = round(term_correct / term_hits, 4)
                metrics["term_hits"] = term_hits
                metrics["term_correct"] = term_correct
                print(f"Term accuracy: {metrics['term_accuracy']} ({term_correct}/{term_hits})")

        qual_examples = {"variant": variant, "best": [], "worst": [], "glossary": []}
        if comet_scores:
            scored = sorted(enumerate(comet_scores), key=lambda item: item[1], reverse=True)
            for idx, score in scored[:10]:
                qual_examples["best"].append(_qual_entry(predictions[idx], float(score)))
            for idx, score in scored[-10:]:
                qual_examples["worst"].append(_qual_entry(predictions[idx], float(score)))

        gloss_examples = [p for p in predictions if p.get("glossary_matches")]
        for pred in gloss_examples[:10]:
            qual_examples["glossary"].append(_qual_entry(pred, pred.get("comet_score", 0)))

        qual_path.parent.mkdir(parents=True, exist_ok=True)
        with qual_path.open("w", encoding="utf-8") as f:
            json.dump(qual_examples, f, ensure_ascii=False, indent=2)
        print(f"Qualitative examples: {qual_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for pred in predictions:
            pred_clean = {key: value for key, value in pred.items() if key != "glossary_matches"}
            f.write(json.dumps(pred_clean, ensure_ascii=False) + "\n")
    print(f"Wrote {len(predictions)} predictions to {output_path}")

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics: {json.dumps(metrics, indent=2)}")

    return {"status": "completed", "variant": variant, "config": config, **metrics}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local S0/S1 EN->JA evaluation")
    parser.add_argument("--variant", default="s0", choices=["s0", "s1"])
    parser.add_argument("--config", default="configs/base.yaml")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(variant=args.variant, config=args.config)


if __name__ == "__main__":
    main()
