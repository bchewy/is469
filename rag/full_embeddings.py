#!/usr/bin/env python3
"""Batch embed chunk JSONL files with skip-if-already-embedded behavior."""

import json
import os
import sys
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer
import torch
from tqdm.auto import tqdm

KB_DIR = Path("kb") if Path("kb").exists() else Path("../kb")
DEFAULT_INPUT_FILES = [
    "eng_jap_chunks.jsonl",
    "style_guide_chunks.jsonl",
    "grammar_chunks.jsonl",
    "gemini_annotated_chunks.jsonl"
]

def resolve_input_files() -> list[Path]:
    configured = os.getenv("EMBED_INPUT_FILES", "").strip()
    if configured:
        names = [name.strip() for name in configured.split(",") if name.strip()]
    else:
        names = DEFAULT_INPUT_FILES

    paths = [KB_DIR / name for name in names]
    missing = [p for p in paths if not p.exists()]
    for p in missing:
        print(f"Warning: input file not found, skipping: {p}")

    return [p for p in paths if p.exists()]


def output_path_for(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_embedded_full.jsonl")


def embed_one_file(
    *,
    model: SentenceTransformer,
    input_path: Path,
    output_path: Path,
    device: str,
    batch_size: int,
    max_chars: int,
) -> None:
    print(f"\n=== Processing {input_path.name} ===")

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Loaded {len(records):,} records from {input_path.name}")
    texts = [r.get("chunk_text", "") for r in records]

    if not texts:
        print("No records found in file. Skipping.")
        return

    # Always prefix document chunks before embedding.
    doc_prefix = "passage: "
    prefixed = 0
    updated_texts = []
    for t in texts:
        if t.lower().startswith(doc_prefix.lower()):
            updated_texts.append(t)
        else:
            updated_texts.append(f"{doc_prefix}{t}")
            prefixed += 1
    texts = updated_texts
    print(f"Applied document prefix to {prefixed:,} chunks: {doc_prefix!r}")

    # Optional CPU-friendly clipping to avoid outlier chunks dominating runtime.
    if max_chars > 0:
        clipped = sum(1 for t in texts if len(t) > max_chars)
        if clipped:
            print(f"Clipping {clipped:,} long chunks to {max_chars} chars for speed.")
            texts = [t[:max_chars] for t in texts]

    total_batches = (len(texts) + batch_size - 1) // batch_size
    print(
        f"Embedding {len(texts):,} chunks in {total_batches:,} batches "
        f"(batch_size={batch_size})..."
    )

    embeddings = []
    start_time = time.perf_counter()
    for start in tqdm(range(0, len(texts), batch_size), desc=f"Embedding {input_path.stem}", unit="batch"):
        batch_texts = texts[start : start + batch_size]
        batch_embeddings = model.encode(
            batch_texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=batch_size,
            device=device,
        )
        embeddings.extend(batch_embeddings)

    elapsed = max(1e-9, time.perf_counter() - start_time)
    print(f"Embedding throughput: {len(texts) / elapsed:.2f} chunks/sec ({elapsed / 60:.1f} min)")

    print("Adding embeddings and saving...")
    for record, embedding in zip(records, embeddings):
        record["embedding"] = embedding.tolist()

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(records):,} embedded records to {output_path}")

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
# There is no official "small" BAAI/bge-m3 model. Default to a lighter multilingual
# sentence-transformer on low-resource machines, while keeping EMBED_MODEL override.
model_name = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
print(f"Loading model: {model_name}")

if device == "cpu":
    # Let torch use multiple CPU threads for faster matmul ops.
    cpu_threads = max(1, min(os.cpu_count() or 1, 16))
    torch.set_num_threads(cpu_threads)
    print(f"CPU threads: {cpu_threads}")

model = SentenceTransformer(model_name, trust_remote_code=True, device=device)

default_max_seq_length = 1024 if device == "cuda" else 384
max_seq_length = int(os.getenv("MAX_SEQ_LENGTH", str(default_max_seq_length)))
model.max_seq_length = max_seq_length
print(f"max_seq_length: {model.max_seq_length}")

default_batch_size = 256 if device == "cuda" else 64
batch_size = int(os.getenv("BATCH_SIZE", str(default_batch_size)))

default_max_chars = 0 if device == "cuda" else 2000
max_chars = int(os.getenv("MAX_CHARS", str(default_max_chars)))

input_paths = resolve_input_files()
if not input_paths:
    print("No input files found to embed. Exiting.")
    sys.exit(0)

skipped = 0
embedded = 0

for input_path in input_paths:
    output_path = output_path_for(input_path)
    if output_path.exists():
        print(f"Skipping {input_path.name}: output already exists at {output_path.name}")
        skipped += 1
        continue

    embed_one_file(
        model=model,
        input_path=input_path,
        output_path=output_path,
        device=device,
        batch_size=batch_size,
        max_chars=max_chars,
    )
    embedded += 1

print(f"\nDone. Embedded files: {embedded}, skipped files: {skipped}")
