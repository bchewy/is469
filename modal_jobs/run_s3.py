from __future__ import annotations

import json
import os
import time
import sys
import gc
from pathlib import Path

import modal

APP_NAME = "enja-run-s3-variant"

ROOT_DIR = Path(__file__).parent.parent
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
        "protobuf>=4.25,<6",
        "sacrebleu>=2.5.0",
        "boto3>=1.37.0",
        "sentence-transformers>=3.3.0",
        "unbabel-comet>=2.2.2",
    )
    # Bundle repo code needed by imports inside the container.
    .add_local_dir(ROOT_DIR / "src", remote_path="/root/src")
    .add_local_dir(ROOT_DIR / "configs", remote_path="/root/configs")
    # Chunk text resolution for S3 Vectors hits (metadata points at *_embedded_full.jsonl lines).
    .add_local_dir(ROOT_DIR / "kb", remote_path="/root/kb")
)

models_volume = modal.Volume.from_name("enja-base-models", create_if_missing=True)
artifacts_volume = modal.Volume.from_name("enja-model-artifacts", create_if_missing=True)
data_volume = modal.Volume.from_name("enja-data", create_if_missing=True)


def _load_yaml(path: str) -> dict:
    # Modal runs in /root by default, but configs may be passed in relative to repo.
    p = Path(path)
    if not p.exists():
        p = Path("/root") / path
    if not p.exists():
        p = Path("/root/configs") / Path(path).name
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    import yaml

    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    # Accept UTF-8 JSONL written by Windows tools that may prepend a BOM.
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
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
        # Character-level BLEU is a safer fallback for Japanese than the default 13a tokenizer.
        bleu_tokenizer = "char"
        bleu = BLEU(tokenize=bleu_tokenizer)
        bleu_result = bleu.corpus_score(hyps, refs)

    chrfpp = CHRF(word_order=2).corpus_score(hyps, refs)

    return {
        "bleu": round(bleu_result.score, 2),
        "bleu_tokenizer": bleu_tokenizer,
        "chrfpp": round(chrfpp.score, 2),
    }


@app.function(
    image=inference_image,
    gpu="A100",
    timeout=60 * 60,
    volumes={
        str(MODELS_DIR): models_volume,
        str(ARTIFACTS_DIR): artifacts_volume,
        str(DATA_DIR): data_volume,
        str(RESULTS_DIR): modal.Volume.from_name(
            "enja-results", create_if_missing=True
        ),
    },
    secrets=[
        modal.Secret.from_name("enja-hf", required_keys=["HF_TOKEN"]),
        # Account B: S3 Vectors API only (not the object S3 bucket for model weights).
        # Omit this secret only when retrieval.enabled is false in config.
        modal.Secret.from_name("enja-s3-vectors"),
    ],
)
def run(config: str) -> dict:
    # Ensure `src.*` imports work with our image add_local_dir layout.
    sys.path.append("/root")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    from src.agents.agentic_rag import detect_translation_error, translate_with_agentic_loop
    from src.eval.s3_eval import (
        build_eval_assets,
        build_retrieval_eval,
        build_terminology_eval,
        compute_comet_metrics,
        compute_error_id_metrics,
        compute_retrieval_metrics,
        compute_terminology_metrics,
    )
    from src.retrieval.s3_vectors_rag import S3VectorsRAGRetriever

    cfg = _load_yaml(config)
    model_cfg = cfg.get("model", {})
    io_cfg = cfg.get("io", {})
    agent_cfg = cfg.get("agent", {})
    retrieval_cfg = cfg.get("retrieval") or {}

    base_model_id = model_cfg.get("base_model_id", "Qwen/Qwen2.5-7B-Instruct")
    model_local_path = model_cfg.get("model_local_path")
    if model_local_path and Path(model_local_path).exists():
        model_path = model_local_path
    else:
        model_path = base_model_id

    adapter_dir = model_cfg.get("adapter_dir")  # expected LoRA adapter directory

    input_path = _resolve_path(io_cfg.get("input_path", "data/splits/test_v1.jsonl"))
    output_path = io_cfg.get("output_path", "results/metrics/s3_outputs.jsonl")

    gen_cfg = cfg.get("generation", {})

    print(f"S3 run config={config}")
    print(f"Model path: {model_path}")
    print(f"Adapter dir: {adapter_dir}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(
        "Generation: "
        f"do_sample={bool(gen_cfg.get('do_sample', False))} "
        f"num_beams={int(gen_cfg.get('num_beams', 1))} "
        f"max_new_tokens={int(gen_cfg.get('max_new_tokens', 512))}"
    )

    rows = _load_jsonl(input_path)
    print(f"Loaded {len(rows)} rows")
    eval_assets = build_eval_assets(rows, retrieval_cfg.get("kb_dir", "/root/kb"))

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

    # If adapter_dir exists and looks like a LoRA adapter, load it.
    if adapter_dir and Path(adapter_dir).exists():
        adapter_config = Path(adapter_dir) / "adapter_config.json"
        if adapter_config.exists():
            print(f"Loading LoRA adapter from {adapter_dir}")
            model = PeftModel.from_pretrained(model, adapter_dir)
            model = model.merge_and_unload()
        else:
            print(
                f"WARNING: adapter_dir exists but adapter_config.json not found: {adapter_dir}. "
                "Running without adapter."
            )
    elif adapter_dir:
        print(f"WARNING: adapter_dir not found: {adapter_dir}. Running without adapter.")

    model.eval()

    retriever: S3VectorsRAGRetriever | None = None
    if retrieval_cfg.get("enabled", True) and retrieval_cfg.get("backend", "s3_vectors") == "s3_vectors":
        if not os.environ.get("VECTORS_AWS_ACCESS_KEY_ID"):
            raise ValueError(
                "S3 Vectors retrieval requires VECTORS_AWS_* credentials (Modal secret enja-s3-vectors). "
                "Or set retrieval.enabled: false. Object S3 (MODELS_*) is a different account and is not used here."
            )
        vbucket = retrieval_cfg.get("vector_bucket_name")
        vindex = retrieval_cfg.get("index_name")
        if not vbucket or not vindex:
            raise ValueError(
                "retrieval.enabled is true but retrieval.vector_bucket_name or retrieval.index_name is missing."
            )
        retriever = S3VectorsRAGRetriever(
            vector_bucket_name=vbucket,
            index_name=vindex,
            region_name=retrieval_cfg.get("region", os.environ.get("AWS_DEFAULT_REGION", "ap-southeast-1")),
            kb_dir=retrieval_cfg.get("kb_dir", "/root/kb"),
            embed_model_name=retrieval_cfg.get("embed_model"),
            top_k=int(retrieval_cfg.get("top_k", 5)),
            max_context_chars=int(retrieval_cfg.get("max_context_chars", 12000)),
        )
        print(
            f"S3 Vectors RAG: bucket={vbucket} index={vindex} "
            f"kb_dir={retrieval_cfg.get('kb_dir', '/root/kb')}"
        )

    predictions: list[dict] = []
    total_coverage = 0.0
    total_latency_ms = 0.0
    total_retrieval_ms = 0.0
    total_rewrite_steps = 0
    total_revision_steps = 0

    for row in rows:
        row_id = row.get("id", "")
        source_en = row.get("source_en", "")
        reference_ja = row.get("target_ja", "")

        retrieval_chunks: list[dict] = []
        retrieval_eval: dict = {
            "expected_target_count": 0,
            "matched_target_count": 0,
            "hit_at_k": False,
            "recall": None,
            "matched_kinds": [],
            "expected_kinds": [],
        }
        retrieval_ms = 0.0
        if retriever is not None:
            retrieval_t0 = time.time()
            context, rchunks = retriever.retrieve(source_en)
            retrieval_ms = round((time.time() - retrieval_t0) * 1000, 1)
            total_retrieval_ms += retrieval_ms
            retrieval_eval = build_retrieval_eval(
                source_en=source_en,
                retrieved_texts=[c.text for c in rchunks],
                assets=eval_assets,
            )
            retrieval_chunks = [
                {
                    "key": c.key,
                    "distance": c.distance,
                    "source_file": c.source_file,
                    "source_line": c.source_line,
                    "text_preview": (c.text[:300] + "…") if len(c.text) > 300 else c.text,
                }
                for c in rchunks
            ]
        else:
            context = ""

        t0 = time.time()
        candidate_ja, coverage_score, trace = translate_with_agentic_loop(
            model=model,
            tokenizer=tokenizer,
            source_en=source_en,
            context=context,
            agent_cfg=agent_cfg,
            gen_cfg=gen_cfg,
        )
        latency_ms = round((time.time() - t0) * 1000, 1)

        total_latency_ms += latency_ms
        total_coverage += coverage_score

        for s in trace:
            if s.step_type == "rewrite":
                total_rewrite_steps += 1
            elif s.step_type == "revision":
                total_revision_steps += 1

        error_check = detect_translation_error(
            model=model,
            tokenizer=tokenizer,
            source_en=source_en,
            candidate_ja=candidate_ja,
            context=context,
            gen_cfg=gen_cfg,
        )
        terminology_eval = build_terminology_eval(
            source_en=source_en,
            prediction_ja=candidate_ja,
            assets=eval_assets,
        )
        gold_error_label = eval_assets.gold_error_by_id.get(row_id)

        predictions.append(
            {
                "id": row_id,
                "source_en": source_en,
                "prediction_ja": candidate_ja,
                "reference_ja": reference_ja,
                "variant": "s3",
                "latency_ms": latency_ms,
                "retrieval_ms": retrieval_ms,
                "coverage_score": coverage_score,
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
                        "critic_coverage_score": t.critic_coverage_score,
                        "critic_has_error": t.critic_has_error,
                        "critic_feedback": t.critic_feedback,
                        "candidate_ja": t.candidate_ja,
                    }
                    for t in trace
                ],
                "retrieval_chunks": retrieval_chunks,
            }
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    has_refs = any(p["reference_ja"] for p in predictions)
    num_samples = max(len(predictions), 1)

    metrics: dict = {
        "variant": "s3",
        "num_samples": len(predictions),
        "avg_latency_ms": round(total_latency_ms / num_samples, 1),
        "avg_retrieval_ms": round(total_retrieval_ms / num_samples, 1),
        "avg_coverage_score": round(total_coverage / num_samples, 4),
        "total_rewrite_steps": total_rewrite_steps,
        "total_revision_steps": total_revision_steps,
    }

    if has_refs:
        metrics.update(_compute_translation_metrics(predictions))
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        try:
            metrics.update(compute_comet_metrics(predictions))
        except Exception as exc:
            metrics["comet_error"] = str(exc)

    metrics.update(compute_retrieval_metrics(predictions))
    metrics.update(compute_terminology_metrics(predictions))
    metrics.update(compute_error_id_metrics(predictions))

    metrics_path = Path(f"results/metrics/s3_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Metrics: {json.dumps(metrics, indent=2, ensure_ascii=False)}")

    return {
        "status": "completed",
        "variant": "s3",
        "config": config,
        **metrics,
    }


@app.local_entrypoint()
def main(config: str = "configs/s3_inference.yaml"):
    print(run.remote(config=config))

