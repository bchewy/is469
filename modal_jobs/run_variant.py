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
    )
    .add_local_dir(
        Path(__file__).parent.parent / "configs",
        remote_path="/root/configs",
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
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_path(p: str) -> str:
    """Try DATA_DIR-prefixed path first, then raw."""
    vol = DATA_DIR / p
    return str(vol) if vol.exists() else p


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

    predictions: list[dict] = []
    total_latency = 0.0

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
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
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
        try:
            from sacrebleu.metrics import BLEU
            bleu = BLEU(tokenize="ja-mecab")
        except Exception:
            from sacrebleu.metrics import BLEU
            bleu = BLEU()

        refs = [[p["reference_ja"] for p in predictions]]
        hyps = [p["prediction_ja"] for p in predictions]
        bleu_result = bleu.corpus_score(hyps, refs)
        metrics["bleu"] = round(bleu_result.score, 2)
        print(f"BLEU: {metrics['bleu']}")

    metrics_path = Path(f"results/metrics/{variant}_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics: {json.dumps(metrics, indent=2)}")

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
