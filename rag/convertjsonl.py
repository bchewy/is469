#!/usr/bin/env python3
"""Convert a FAISS index into JSONL, optionally merging row-wise metadata."""

import argparse
import importlib
import json
import sys
from pathlib import Path


def import_faiss_library():
	"""Import the real FAISS package even when a local faiss.py exists."""
	script_dir = Path(__file__).resolve().parent
	removed_entries: list[tuple[int, str]] = []

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
			if existing_file and Path(existing_file).resolve().name == "faiss.py":
				del sys.modules["faiss"]
		return importlib.import_module("faiss")
	except ModuleNotFoundError as exc:
		raise SystemExit(
			"FAISS package is not installed. Install one of: faiss-cpu or faiss-gpu."
		) from exc
	finally:
		for idx, entry in sorted(removed_entries, key=lambda item: item[0]):
			sys.path.insert(idx, entry)


faiss = import_faiss_library()


def load_metadata(metadata_path: Path | None) -> list[dict] | None:
	if metadata_path is None:
		return None
	if not metadata_path.exists():
		raise SystemExit(f"Metadata file not found: {metadata_path}")

	rows: list[dict] = []
	with open(metadata_path, "r", encoding="utf-8") as f:
		for line in f:
			rows.append(json.loads(line))
	return rows


def convert_index_to_jsonl(
	*,
	index_path: Path,
	output_path: Path,
	metadata_path: Path | None,
	batch_size: int,
) -> None:
	if not index_path.exists():
		raise SystemExit(f"Index file not found: {index_path}")

	print(f"Loading index: {index_path}")
	index = faiss.read_index(str(index_path))
	total = index.ntotal
	print(f"Vectors in index: {total:,}")

	if total == 0:
		raise SystemExit("Index has no vectors.")

	metadata_rows = load_metadata(metadata_path)
	if metadata_rows is not None and len(metadata_rows) != total:
		raise SystemExit(
			"Metadata row count does not match index size: "
			f"metadata={len(metadata_rows):,}, index={total:,}"
		)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	with open(output_path, "w", encoding="utf-8") as out:
		for start in range(0, total, batch_size):
			end = min(start + batch_size, total)
			for i in range(start, end):
				try:
					vector = index.reconstruct(i)
				except Exception as exc:
					raise SystemExit(
						"This FAISS index does not support vector reconstruction. "
						"Use a reconstructable index (for example IndexFlatIP/IndexFlatL2), "
						"or export from the original embeddings before indexing. "
						f"Failed at row {i}."
					) from exc

				row = {
					"id": i,
					"embedding": vector.tolist(),
				}
				if metadata_rows is not None:
					row.update(metadata_rows[i])
				out.write(json.dumps(row, ensure_ascii=False) + "\n")

			print(f"Wrote rows {start:,} to {end - 1:,}")

	print(f"Done. JSONL written to: {output_path}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Convert FAISS index vectors to JSONL with optional metadata merge."
	)
	parser.add_argument(
		"--index",
		type=Path,
		default=Path("rag/knowledge_base.index"),
		help="Path to FAISS .index file",
	)
	parser.add_argument(
		"--metadata",
		type=Path,
		default=Path("rag/knowledge_base_metadata.jsonl"),
		help="Optional metadata JSONL with one row per vector",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("rag/knowledge_base_vectors.jsonl"),
		help="Output JSONL path",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=512,
		help="Number of vectors processed per progress chunk",
	)
	parser.add_argument(
		"--no-metadata",
		action="store_true",
		help="Export only id + embedding without metadata",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	metadata_path = None if args.no_metadata else args.metadata
	convert_index_to_jsonl(
		index_path=args.index,
		output_path=args.output,
		metadata_path=metadata_path,
		batch_size=max(1, args.batch_size),
	)


if __name__ == "__main__":
	main()
