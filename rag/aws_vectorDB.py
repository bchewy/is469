import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.aws_profiles import s3vectors_client

# Vector-bucket account (use VECTORS_AWS_*; see src/utils/aws_profiles.py)
_region = os.environ.get("VECTORS_AWS_DEFAULT_REGION", "ap-southeast-1")
client = s3vectors_client(region_name=_region)

BUCKET_NAME = 'is469-genai-grp-project'
INDEX_NAME = 'rag-vector-2'
BATCH_SIZE = 10
MAX_METADATA_BYTES = 1900

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
KB_DIR = WORKSPACE_ROOT / "kb"

list_files = [
    "style_guide_chunks_embedded_full_vectors.jsonl",
    "eng_jap_chunks_embedded_full_vectors.jsonl",
    "gemini_annotated_chunks_embedded_full_vectors.jsonl",
    "grammar_chunks_embedded_full_vectors.jsonl",
]


def flush_batch(vectors_batch: list[dict]) -> None:
    if not vectors_batch:
        return
    client.put_vectors(
        vectorBucketName=BUCKET_NAME,
        indexName=INDEX_NAME,
        vectors=vectors_batch,
    )


def load_metadata_rows(metadata_path: Path) -> list[dict]:
    rows: list[dict] = []
    if not metadata_path.exists():
        print(f"Warning: metadata file not found, using minimal metadata: {metadata_path}")
        return rows

    with open(metadata_path, "r", encoding="utf-8") as file:
        for line in file:
            rows.append(json.loads(line))
    return rows


def sanitize_metadata(raw: dict, source_file: str) -> dict:
    metadata = {
        "source_file": str(raw.get("source_file", source_file)),
        "source_line": int(raw.get("source_line", -1)),
        "chunk_id": str(raw.get("chunk_id", ""))[:120],
    }

    # S3 Vectors filterable metadata has a strict size cap.
    metadata_size = len(json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    if metadata_size > MAX_METADATA_BYTES:
        metadata = {
            "source_file": metadata["source_file"],
            "source_line": metadata["source_line"],
        }
    return metadata


vectors_batch: list[dict] = []
seen_keys_in_batch: set[str] = set()

for file_name in list_files:
    input_path = KB_DIR / file_name
    if not input_path.exists():
        print(f"Warning: file not found, skipping: {input_path}")
        continue

    source_prefix = input_path.stem.replace("_vectors", "")
    metadata_path = KB_DIR / f"{source_prefix}_metadata.jsonl"
    metadata_rows = load_metadata_rows(metadata_path)

    print(f"Loading vectors from: {input_path}")

    with open(input_path, 'r', encoding='utf-8') as file:
        for line_idx, line in enumerate(file):
            data = json.loads(line)
            record_id = str(data.get('id', ''))
            if not record_id:
                continue

            metadata_row: dict = {}
            try:
                record_index = int(record_id)
            except ValueError:
                record_index = line_idx

            if 0 <= record_index < len(metadata_rows):
                metadata_row = metadata_rows[record_index]

            # Namespace keys by source file to avoid duplicate IDs across files.
            key = f"{source_prefix}:{record_id}"

            # Ensure no duplicate keys in a single PutVectors request.
            if key in seen_keys_in_batch:
                continue

            vector_record = {
                'key': key,
                'data': {'float32': data['embedding']},
                'metadata': sanitize_metadata(metadata_row, source_prefix),
            }
            vectors_batch.append(vector_record)
            seen_keys_in_batch.add(key)

            if len(vectors_batch) >= BATCH_SIZE:
                flush_batch(vectors_batch)
                vectors_batch = []
                seen_keys_in_batch = set()

flush_batch(vectors_batch)

print("Ingestion into S3 Vector Bucket complete!")
