#!/usr/bin/env python3
"""Build one FAISS index from multiple embedded JSONL files."""

import json
import importlib
import os
import sys
from pathlib import Path

import numpy as np

KB_DIR = Path("kb") if Path("kb").exists() else Path("../kb")
DEFAULT_INPUT_FILES = [
	"eng_jap_chunks_embedded_full.jsonl",
	"style_guide_chunks_embedded_full.jsonl",
	"grammar_chunks_embedded_full.jsonl",
    "gemini_annotated_chunks_embedded_full.jsonl"
]


def import_faiss_library():
	"""Import the real FAISS package even when this file is named faiss.py."""
	script_dir = Path(__file__).resolve().parent
	removed_entries: list[tuple[int, str]] = []

	# Remove any path entry that resolves to this script directory, including '' and '.'.
	for idx in range(len(sys.path) - 1, -1, -1):
		entry = sys.path[idx]
		entry_path = Path(entry or ".").resolve()
		if entry_path == script_dir:
			removed_entries.append((idx, entry))
			del sys.path[idx]

	try:
		existing = sys.modules.get("faiss")
		if existing is not None:
			existing_file = getattr(existing, "__file__", "")
			if existing_file and Path(existing_file).resolve() == Path(__file__).resolve():
				del sys.modules["faiss"]
		return importlib.import_module("faiss")
	except ModuleNotFoundError as exc:
		raise SystemExit(
			"FAISS package is not installed in this environment. "
			"Install one of: faiss-cpu or faiss-gpu."
		) from exc
	finally:
		# Restore removed sys.path entries at their original positions.
		for idx, entry in sorted(removed_entries, key=lambda item: item[0]):
			sys.path.insert(idx, entry)


faiss_lib = import_faiss_library()


def resolve_input_files() -> list[Path]:
	configured = os.getenv("FAISS_INPUT_FILES", "").strip()
	if configured:
		names = [name.strip() for name in configured.split(",") if name.strip()]
	else:
		names = DEFAULT_INPUT_FILES

	paths = [KB_DIR / name for name in names]
	missing = [p for p in paths if not p.exists()]
	for path in missing:
		print(f"Warning: input file not found, skipping: {path}")

	return [p for p in paths if p.exists()]


def load_embeddings_from_file(path: Path) -> tuple[list[list[float]], list[dict]]:
	vectors: list[list[float]] = []
	metadata: list[dict] = []

	with open(path, "r", encoding="utf-8") as f:
		for line_no, line in enumerate(f, start=1):
			record = json.loads(line)
			embedding = record.get("embedding")
			if embedding is None:
				continue

			vectors.append(embedding)
			metadata.append(
				{
					"source_file": path.name,
					"source_line": line_no,
					"chunk_id": record.get("chunk_id"),
					"chunk_text": record.get("chunk_text", ""),
				}
			)

	print(f"Loaded {len(vectors):,} embeddings from {path.name}")
	return vectors, metadata


def l2_normalize_inplace(vectors: np.ndarray) -> None:
	"""L2-normalize each row in-place for cosine-style inner product search."""
	norms = np.linalg.norm(vectors, axis=1, keepdims=True)
	# Avoid division by zero for empty/zero vectors.
	norms = np.maximum(norms, 1e-12)
	vectors /= norms


def main() -> None:
	input_files = resolve_input_files()
	if not input_files:
		raise SystemExit("No embedded files found. Set FAISS_INPUT_FILES or place files in kb/.")

	all_vectors: list[list[float]] = []
	all_metadata: list[dict] = []

	for file_path in input_files:
		vectors, metadata = load_embeddings_from_file(file_path)
		all_vectors.extend(vectors)
		all_metadata.extend(metadata)

	if not all_vectors:
		raise SystemExit("No embeddings were loaded from the selected files.")

	embeddings_array = np.array(all_vectors, dtype="float32")
	if embeddings_array.ndim != 2:
		raise SystemExit(f"Embeddings must be 2D, got shape={embeddings_array.shape}")

	dimension = embeddings_array.shape[1]
	print(f"Creating FAISS index with dimension: {dimension}")

	# Normalize so IndexFlatIP works as cosine similarity.
	l2_normalize_inplace(embeddings_array)
	index = faiss_lib.IndexFlatIP(dimension)
	index.add(embeddings_array)

	print(f"Successfully added {index.ntotal:,} vectors to the database.")

	index_path = Path(os.getenv("FAISS_INDEX_PATH", "./kb/knowledge_base.index"))
	metadata_path = Path(os.getenv("FAISS_METADATA_PATH", "./kb/knowledge_base_metadata.jsonl"))

	faiss_lib.write_index(index, str(index_path))
	print(f"Saved FAISS index to {index_path}")

	with open(metadata_path, "w", encoding="utf-8") as f:
		for row in all_metadata:
			f.write(json.dumps(row, ensure_ascii=False) + "\n")
	print(f"Saved metadata ({len(all_metadata):,} rows) to {metadata_path}")


if __name__ == "__main__":
	main()