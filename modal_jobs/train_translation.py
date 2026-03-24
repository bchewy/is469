from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal
import yaml

APP_NAME = "enja-train-translation"

MODELS_DIR = Path("/models")
ARTIFACTS_DIR = Path("/artifacts")
DATA_DIR = Path("/data")

app = modal.App(APP_NAME)

training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.5.1",
        "transformers>=4.46.0",
        "peft>=0.14.0",
        "datasets>=3.2.0",
        "accelerate>=1.2.0",
        "trl>=0.13.0",
        "bitsandbytes>=0.45.0",
        "pyyaml>=6.0.2",
        "sentencepiece>=0.2.0",
        "protobuf>=5.29.0",
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


def _format_chat(source_en: str, target_ja: str | None = None) -> list[dict]:
    """Build chat-format messages for SFT."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": source_en},
    ]
    if target_ja is not None:
        messages.append({"role": "assistant", "content": target_ja})
    return messages


def _load_jsonl_dataset(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@app.function(
    image=training_image,
    gpu="A100",
    timeout=60 * 120,
    volumes={
        str(MODELS_DIR): models_volume,
        str(ARTIFACTS_DIR): artifacts_volume,
        str(DATA_DIR): data_volume,
    },
    secrets=[modal.Secret.from_name("enja-hf", required_keys=["HF_TOKEN"])],
)
def train(config: str) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTTrainer, SFTConfig

    cfg = _load_yaml(config)
    seed = cfg.get("project", {}).get("seed", 42)
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})

    base_model_id = model_cfg.get("base_model_id", "Qwen/Qwen2.5-7B-Instruct")
    output_adapter_dir = model_cfg.get("output_adapter_dir", "/artifacts/adapters/translation")

    local_model_path = model_cfg.get("model_local_path")
    if local_model_path and Path(local_model_path).exists():
        model_path = local_model_path
        print(f"Using local model: {model_path}")
    else:
        model_path = base_model_id
        print(f"Using HF model: {model_path}")

    epochs = train_cfg.get("epochs", 2)
    lr = train_cfg.get("lr", 2e-4)
    lora_rank = train_cfg.get("lora_rank", 16)
    batch_size = train_cfg.get("batch_size", 8)
    max_seq_len = train_cfg.get("max_seq_len", 2048)

    train_path = data_cfg.get("train_path", "data/splits/train_v1.jsonl")
    dev_path = data_cfg.get("dev_path", "data/splits/dev_v1.jsonl")

    for p in [train_path, dev_path]:
        data_vol_path = DATA_DIR / p
        if data_vol_path.exists():
            pass
        elif Path(p).exists():
            pass
        else:
            print(f"WARNING: data file not found at {p} or {data_vol_path}")

    def resolve_data(p: str) -> str:
        vol = DATA_DIR / p
        return str(vol) if vol.exists() else p

    train_rows = _load_jsonl_dataset(resolve_data(train_path))
    dev_rows = _load_jsonl_dataset(resolve_data(dev_path))
    print(f"Train rows: {len(train_rows)}, Dev rows: {len(dev_rows)}")

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
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def format_row(row: dict) -> dict:
        messages = _format_chat(row["source_en"], row.get("target_ja"))
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    train_ds = Dataset.from_list(train_rows).map(format_row)
    dev_ds = Dataset.from_list(dev_rows).map(format_row)

    output_dir = str(Path(output_adapter_dir) / "checkpoints")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=max(1, 32 // batch_size),
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        seed=seed,
        report_to="none",
        max_grad_norm=0.3,
        optim="paged_adamw_8bit",
        max_length=max_seq_len,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        args=sft_config,
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    final_adapter_dir = str(Path(output_adapter_dir) / "final")
    Path(final_adapter_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)
    print(f"Adapter saved to {final_adapter_dir}")

    artifacts_volume.commit()

    eval_results = trainer.evaluate()

    result = {
        "status": "completed",
        "job": "train_translation",
        "config": config,
        "base_model": base_model_id,
        "adapter_path": final_adapter_dir,
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "epochs": epochs,
        "lora_rank": lora_rank,
        "lr": lr,
        "elapsed_seconds": round(elapsed, 1),
        "eval_loss": eval_results.get("eval_loss"),
    }
    print(json.dumps(result, indent=2))
    return result


@app.local_entrypoint()
def main(config: str = "configs/finetune_translation.yaml"):
    print(train.remote(config=config))
