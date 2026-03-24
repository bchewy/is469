from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal
import yaml

APP_NAME = "enja-run-variant"

MODELS_DIR = Path("/models")
ARTIFACTS_DIR = Path("/artifacts")
DATA_DIR = Path("/data")
RESULTS_DIR = Path("/results")

app = modal.App(APP_NAME)

inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.5.1",
        "transformers>=4.46.0",
        "peft>=0.14.0",
        "accelerate>=1.2.0",
        "bitsandbytes>=0.45.0",
        "pyyaml>=6.0.2",
        "sentencepiece>=0.2.0",
        "protobuf>=5.29.0",
        "sacrebleu>=2.5.0",
        "unbabel-comet>=2.2.0",
    )
    .add_local_dir(
        Path(__file__).parent.parent / "configs",
        remote_path="/root/configs",
    )
    .add_local_dir(
        Path(__file__).parent.parent / "kb",
        remote_path="/root/kb",
    )
)

models_volume = modal.Volume.from_name("enja-base-models", create_if_missing=True)
artifacts_volume = modal.Volume.from_name("enja-model-artifacts", create_if_missing=True)
data_volume = modal.Volume.from_name("enja-data", create_if_missing=True)


def _load_yaml(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        p = Path("/root") / path
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


SYSTEM_PROMPT = (
    "You are a professional English-to-Japanese translator. "
    "Translate the following English text into natural, fluent Japanese. "
    "Preserve the original meaning and tone."
)


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    # Accept UTF-8 JSONL written by Windows tools that may prepend a BOM.
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_path(p: str) -> str:
    """Try DATA_DIR-prefixed path first, then raw."""
    vol = DATA_DIR / p
    return str(vol) if vol.exists() else p


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


def _load_glossary() -> list[tuple[str, str]]:
    """Load glossary CSV and return (en_term, ja_term) pairs."""
    import csv
    glossary_paths = [
        Path("/root/kb/glossary.csv"),
        Path("kb/glossary.csv"),
    ]
    for gp in glossary_paths:
        if gp.exists():
            pairs = []
            with gp.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    en = row.get("source_term_en", "").strip().lower()
                    ja = row.get("approved_ja", "").strip()
                    if en and ja:
                        pairs.append((en, ja))
            print(f"Loaded {len(pairs)} glossary terms from {gp}")
            return pairs
    print("No glossary file found")
    return []


def _qual_entry(pred: dict, score: float) -> dict:
    return {
        "id": pred.get("id", ""),
        "source_en": pred["source_en"],
        "prediction_ja": pred["prediction_ja"],
        "reference_ja": pred.get("reference_ja", ""),
        "comet_score": round(score, 4),
        "glossary_matches": pred.get("glossary_matches", []),
    }


@app.function(
    image=inference_image,
    gpu="A100",
    timeout=60 * 60,
    volumes={
        str(MODELS_DIR): models_volume,
        str(ARTIFACTS_DIR): artifacts_volume,
        str(DATA_DIR): data_volume,
        str(RESULTS_DIR): modal.Volume.from_name("enja-results", create_if_missing=True),
    },
    secrets=[modal.Secret.from_name("enja-hf", required_keys=["HF_TOKEN"])],
)
def run(variant: str, config: str) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    cfg = _load_yaml(config)
    model_cfg = cfg.get("model", {})
    io_cfg = cfg.get("io", {})
    gen_cfg = cfg.get("generation", {})

    base_model_id = model_cfg.get("base_model_id", "Qwen/Qwen2.5-7B-Instruct")
    local_model_path = model_cfg.get("model_local_path")

    if local_model_path and Path(local_model_path).exists():
        model_path = local_model_path
    else:
        model_path = base_model_id

    input_path = _resolve_path(io_cfg.get("input_path", "data/splits/test_v1.jsonl"))
    output_path = io_cfg.get("output_path", f"results/metrics/{variant}_outputs.jsonl")

    adapter_dir = model_cfg.get("adapter_dir", "/artifacts/adapters/translation/final")

    print(f"Variant: {variant}")
    print(f"Model: {model_path}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(
        "Generation: "
        f"do_sample={bool(gen_cfg.get('do_sample', False))} "
        f"num_beams={int(gen_cfg.get('num_beams', 1))} "
        f"max_new_tokens={int(gen_cfg.get('max_new_tokens', 512))}"
    )

    rows = _load_jsonl(input_path)
    print(f"Loaded {len(rows)} eval rows")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
        torch_dtype=torch.bfloat16,
    )

    if variant == "s1" and Path(adapter_dir).exists():
        print(f"Loading LoRA adapter from {adapter_dir}")
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()
    elif variant == "s1":
        print(f"WARNING: adapter_dir {adapter_dir} not found, running without adapter")

    model.eval()

    seed = cfg.get("project", {}).get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    predictions: list[dict] = []
    total_latency = 0.0
    do_sample = bool(gen_cfg.get("do_sample", False))
    num_beams = int(gen_cfg.get("num_beams", 1))
    max_new_tokens = int(gen_cfg.get("max_new_tokens", 512))
    temperature = float(gen_cfg.get("temperature", 0.1))
    top_p = float(gen_cfg.get("top_p", 0.95))

    for row in rows:
        source_en = row.get("source_en", "")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": source_en},
        ]

        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

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
        latency_ms = round((time.time() - t0) * 1000, 1)
        total_latency += latency_ms

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        translation = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        pred = {
            "id": row.get("id", ""),
            "source_en": source_en,
            "prediction_ja": translation,
            "reference_ja": row.get("target_ja", ""),
            "variant": variant,
            "latency_ms": latency_ms,
        }
        predictions.append(pred)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Wrote {len(predictions)} predictions to {output_path}")

    has_refs = any(p["reference_ja"] for p in predictions)
    metrics: dict = {
        "variant": variant,
        "num_samples": len(predictions),
        "avg_latency_ms": round(total_latency / max(len(predictions), 1), 1),
    }

    if has_refs:
        refs = [[p["reference_ja"] for p in predictions]]
        hyps = [p["prediction_ja"] for p in predictions]
        srcs = [p["source_en"] for p in predictions]

        metrics.update(_compute_translation_metrics(predictions))
        print(
            f"BLEU ({metrics['bleu_tokenizer']}): {metrics['bleu']}, "
            f"chrF++: {metrics['chrfpp']}"
        )

        # ── COMET ────────────────────────────────────────────────────
        try:
            from comet import download_model, load_from_checkpoint
            comet_path = download_model("Unbabel/wmt22-comet-da")
            comet_model = load_from_checkpoint(comet_path)
            comet_data = [
                {"src": s, "mt": h, "ref": r}
                for s, h, r in zip(srcs, hyps, [p["reference_ja"] for p in predictions])
            ]
            comet_output = comet_model.predict(comet_data, batch_size=32, gpus=1)
            comet_scores = comet_output.scores
            metrics["comet"] = round(comet_output.system_score, 4)
            print(f"COMET: {metrics['comet']}")
            for i, p in enumerate(predictions):
                p["comet_score"] = round(comet_scores[i], 4)
        except Exception as exc:
            print(f"COMET evaluation failed: {exc}")
            comet_scores = None

        # ── Terminology accuracy ─────────────────────────────────────
        glossary = _load_glossary()
        if glossary:
            term_hits = 0
            term_correct = 0
            for p in predictions:
                src_lower = p["source_en"].lower()
                pred_ja = p["prediction_ja"]
                matched_terms = []
                for en_term, ja_term in glossary:
                    if en_term in src_lower:
                        term_hits += 1
                        if ja_term in pred_ja:
                            term_correct += 1
                            matched_terms.append((en_term, ja_term, True))
                        else:
                            matched_terms.append((en_term, ja_term, False))
                p["glossary_matches"] = matched_terms
            if term_hits > 0:
                metrics["term_accuracy"] = round(term_correct / term_hits, 4)
                metrics["term_hits"] = term_hits
                metrics["term_correct"] = term_correct
                print(f"Term accuracy: {metrics['term_accuracy']} ({term_correct}/{term_hits})")

        # ── Qualitative examples ─────────────────────────────────────
        qual_dir = Path(f"results/qualitative_examples")
        qual_dir.mkdir(parents=True, exist_ok=True)
        qual_examples = {"variant": variant, "best": [], "worst": [], "glossary": []}

        if comet_scores:
            scored = sorted(enumerate(comet_scores), key=lambda x: x[1], reverse=True)
            for idx, score in scored[:10]:
                qual_examples["best"].append(_qual_entry(predictions[idx], score))
            for idx, score in scored[-10:]:
                qual_examples["worst"].append(_qual_entry(predictions[idx], score))

        gloss_examples = [p for p in predictions if p.get("glossary_matches")]
        for p in gloss_examples[:10]:
            qual_examples["glossary"].append(_qual_entry(p, p.get("comet_score", 0)))

        qual_path = qual_dir / f"{variant}.json"
        with qual_path.open("w", encoding="utf-8") as f:
            json.dump(qual_examples, f, ensure_ascii=False, indent=2)
        print(f"Qualitative examples: {qual_path}")

    # ── Write final metrics ──────────────────────────────────────
    metrics_path = Path(f"results/metrics/{variant}_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics: {json.dumps(metrics, indent=2)}")

    with out.open("w", encoding="utf-8") as f:
        for p in predictions:
            p_clean = {k: v for k, v in p.items() if k != "glossary_matches"}
            f.write(json.dumps(p_clean, ensure_ascii=False) + "\n")

    result = {
        "status": "completed",
        "variant": variant,
        "config": config,
        **metrics,
    }
    return result


@app.local_entrypoint()
def main(variant: str = "s0", config: str = "configs/base.yaml"):
    print(run.remote(variant=variant, config=config))
