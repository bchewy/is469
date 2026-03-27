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


def load_manifest(manifest_path: Path) -> list[dict]:
	if not manifest_path.exists():
		raise SystemExit(f"Manifest file not found: {manifest_path}")

	with open(manifest_path, "r", encoding="utf-8") as f:
		data = json.load(f)

	if not isinstance(data, list):
		raise SystemExit(f"Manifest must be a JSON array: {manifest_path}")
	return data


def infer_metadata_for_index(index_path: Path) -> Path:
	return index_path.with_name(f"{index_path.stem}_metadata.jsonl")


def resolve_path(path_str: str, *, base_dir: Path | None = None) -> Path:
	"""Resolve possibly-relative paths against an optional base directory."""
	path = Path(path_str)
	if path.is_absolute():
		return path
	if base_dir is None:
		return path

	# Try common relative roots first to avoid duplicating path prefixes like kb/kb/.
	candidates = [
		path,
		base_dir / path,
		base_dir.parent / path,
	]
	for candidate in candidates:
		if candidate.exists():
			return candidate

	# Fall back to manifest-relative when no candidate currently exists.
	return base_dir / path


def convert_many_indexes(
	*,
	jobs: list[dict],
	output_dir: Path,
	batch_size: int,
	no_metadata: bool,
) -> None:
	if not jobs:
		raise SystemExit("No index conversion jobs were provided.")

	output_dir.mkdir(parents=True, exist_ok=True)

	failures: list[tuple[str, str]] = []

	for job in jobs:
		index_path = Path(job["index_path"])
		metadata_path = None
		if not no_metadata:
			metadata_value = job.get("metadata_path")
			metadata_path = Path(metadata_value) if metadata_value else infer_metadata_for_index(index_path)

		output_path = output_dir / f"{index_path.stem}_vectors.jsonl"
		print(f"\nConverting {index_path} -> {output_path}")
		try:
			convert_index_to_jsonl(
				index_path=index_path,
				output_path=output_path,
				metadata_path=metadata_path,
				batch_size=batch_size,
			)
		except SystemExit as exc:
			failures.append((str(index_path), str(exc)))
			print(f"Skipping failed index {index_path}: {exc}")

	if failures:
		print("\nBatch conversion completed with failures:")
		for index_name, reason in failures:
			print(f"- {index_name}: {reason}")
		raise SystemExit(f"{len(failures)} index file(s) failed during batch conversion.")


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
		"--indexes",
		type=Path,
		nargs="+",
		help="Convert multiple FAISS .index files in one run",
	)
	parser.add_argument(
		"--manifest",
		type=Path,
		help=(
			"Path to knowledge_base_indexes.json manifest generated by rag/faiss.py; "
			"used to batch-convert all listed indexes"
		),
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
		"--output-dir",
		type=Path,
		default=Path("kb"),
		help="Output directory for batch conversion mode",
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
	batch_size = max(1, args.batch_size)

	if args.manifest is not None:
		manifest_path = args.manifest
		rows = load_manifest(manifest_path)
		manifest_dir = manifest_path.parent.resolve()
		jobs = []
		for row in rows:
			if not isinstance(row, dict) or "index_path" not in row:
				raise SystemExit("Each manifest row must include at least: index_path")
			index_path = resolve_path(str(row["index_path"]), base_dir=manifest_dir)
			metadata_value = row.get("metadata_path")
			metadata_path = (
				resolve_path(str(metadata_value), base_dir=manifest_dir)
				if metadata_value is not None
				else None
			)
			jobs.append(
				{
					"index_path": str(index_path),
					"metadata_path": str(metadata_path) if metadata_path is not None else None,
				}
			)
		convert_many_indexes(
			jobs=jobs,
			output_dir=args.output_dir,
			batch_size=batch_size,
			no_metadata=args.no_metadata,
		)
		return

	if args.indexes:
		jobs = [{"index_path": str(index_path)} for index_path in args.indexes]
		convert_many_indexes(
			jobs=jobs,
			output_dir=args.output_dir,
			batch_size=batch_size,
			no_metadata=args.no_metadata,
		)
		return

	metadata_path = None if args.no_metadata else args.metadata
	convert_index_to_jsonl(
		index_path=args.index,
		output_path=args.output,
		metadata_path=metadata_path,
		batch_size=batch_size,
	)


if __name__ == "__main__":
	main()
