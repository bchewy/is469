#!/usr/bin/env python3
"""Batch embed eng_jap_chunks.jsonl with BGE-M3 (optimized for large-scale processing)."""

import json
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch

KB_DIR = Path("kb") if Path("kb").exists() else Path("../kb")
input_path = KB_DIR / "eng_jap_chunks.jsonl"
output_path = KB_DIR / "eng_jap_chunks_embedded.jsonl"

if not input_path.exists():
    print(f"Error: {input_path} not found")
    sys.exit(1)

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Loading BGE-M3 model...")
model = SentenceTransformer("BAAI/bge-m3", trust_remote_code=True)
model.to(device)

# Read and process
print(f"Loading {input_path.name}...")
records = []
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

print(f"Loaded {len(records):,} records. Embedding...")

texts = [r.get("chunk_text", "") for r in records]
batch_size = 256 if device == "cuda" else 16

embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    show_progress_bar=True,
    batch_size=batch_size,
    device=device,
)

# Add embeddings and save
print(f"Adding embeddings and saving...")
for record, embedding in zip(records, embeddings):
    record["embedding"] = embedding.tolist()

with open(output_path, "w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"✓ Saved {len(records):,} embedded records to {output_path}")
