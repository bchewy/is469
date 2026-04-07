#!/usr/bin/env python3
"""Standalone QLoRA fine-tuning script for EN-JA translation.

Runs on any machine with a GPU — no Modal dependency.

Usage:
    python scripts/train_local.py --config configs/finetune_translation.yaml
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from trl import SFTTrainer, SFTConfig

SYSTEM_PROMPT = (
    "You are a professional English-to-Japanese translator. "
    "Translate the following English text into natural, fluent Japanese. "
    "Preserve the original meaning and tone."
)


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def format_chat(source_en: str, target_ja: str | None = None) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": source_en},
    ]
    if target_ja is not None:
        messages.append({"role": "assistant", "content": target_ja})
    return messages


def main():
    import argparse

    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for EN-JA translation")
    parser.add_argument("--config", default="configs/finetune_translation.yaml")
    parser.add_argument("--output-dir", default=None, help="Override adapter output directory")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed = cfg.get("project", {}).get("seed", 42)
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})

    base_model_id = model_cfg.get("base_model_id", "Qwen/Qwen2.5-7B-Instruct")
    output_adapter_dir = args.output_dir or model_cfg.get("output_adapter_dir", "outputs/adapter")

    local_model_path = model_cfg.get("model_local_path")
    if local_model_path and Path(local_model_path).exists():
        model_path = local_model_path
        print(f"Using local model: {model_path}")
    else:
        model_path = base_model_id
        print(f"Using HF model: {model_path}")

    epochs = train_cfg.get("epochs", 3)
    lr = float(train_cfg.get("lr", 3e-5))
    lora_rank = train_cfg.get("lora_rank", 64)
    lora_alpha = train_cfg.get("lora_alpha", lora_rank)
    batch_size = train_cfg.get("batch_size", 8)
    max_seq_len = train_cfg.get("max_seq_len", 2048)
    early_stopping_patience = train_cfg.get("early_stopping_patience", 0)
    eval_steps = train_cfg.get("eval_steps", 500)

    train_path = data_cfg.get("train_path", "data/splits/train_v1.jsonl")
    dev_path = data_cfg.get("dev_path", "data/splits/dev_v1.jsonl")

    train_rows = load_jsonl(train_path)
    dev_rows = load_jsonl(dev_path)
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
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def format_row(row: dict) -> dict:
        return {"messages": format_chat(row["source_en"], row.get("target_ja"))}

    train_ds = Dataset.from_list(train_rows).map(format_row)
    dev_ds = Dataset.from_list(dev_rows).map(format_row)

    checkpoint_dir = str(Path(output_adapter_dir) / "checkpoints")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    use_early_stopping = early_stopping_patience > 0

    sft_config = SFTConfig(
        output_dir=checkpoint_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=max(1, 32 // batch_size),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=use_early_stopping,
        metric_for_best_model="eval_loss" if use_early_stopping else None,
        greater_is_better=False if use_early_stopping else None,
        bf16=True,
        seed=seed,
        report_to="none",
        max_grad_norm=0.3,
        optim="adamw_torch_fused",
        max_length=max_seq_len,
        packing=False,
    )

    callbacks = []
    if use_early_stopping:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        args=sft_config,
        callbacks=callbacks if callbacks else None,
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    final_dir = str(Path(output_adapter_dir) / "final")
    Path(final_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Adapter saved to {final_dir}")

    eval_results = trainer.evaluate()

    result = {
        "status": "completed",
        "base_model": base_model_id,
        "adapter_path": final_dir,
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "epochs": epochs,
        "lora_rank": lora_rank,
        "lr": lr,
        "elapsed_seconds": round(elapsed, 1),
        "eval_loss": eval_results.get("eval_loss"),
    }
    print(json.dumps(result, indent=2))

    results_path = Path(output_adapter_dir) / "train_results.json"
    with results_path.open("w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
