from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.aws_profiles import s3vectors_client


@dataclass
class RetrievedChunk:
    key: str
    distance: float | None
    source_file: str
    source_line: int
    text: str


def _read_jsonl_line(path: Path, line_number: int) -> dict[str, Any] | None:
    """Return the JSON object at 1-based line_number, or None."""
    if line_number < 1 or not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if i == line_number:
                line = line.strip()
                if not line:
                    return None
                return json.loads(line)
    return None


def _guess_kb_paths(kb_dir: Path, source_file: str) -> list[Path]:
    stem = source_file.strip()
    # Strip .jsonl extension if already present in metadata source_file value
    if stem.endswith(".jsonl"):
        stem = stem[:-6]
    candidates = [
        kb_dir / f"{stem}.jsonl",
        kb_dir / f"{stem}_vectors.jsonl",
        kb_dir / f"{stem}_embedded_full.jsonl",
    ]
    if stem.endswith("_full"):
        candidates.append(kb_dir / f"{stem.replace('_full', '')}_embedded_full.jsonl")
    return [p for p in candidates if p.is_file()]


def _chunk_text_from_record(record: dict[str, Any]) -> str:
    return str(
        record.get("chunk_text")
        or record.get("text")
        or record.get("content")
        or ""
    ).strip()


def format_context(chunks: list[RetrievedChunk], *, max_chars: int) -> str:
    # Sort: glossary and TM entries first, then everything else
    def priority(c: RetrievedChunk) -> int:
        sf = c.source_file.lower()
        if "glossary" in sf:
            return 0
        if "translation_memory" in sf:
            return 1
        return 2

    sorted_chunks = sorted(chunks, key=priority)
    parts: list[str] = []
    for ch in sorted_chunks:
        label = f"[{ch.source_file} L{ch.source_line}]"
        parts.append(f"{label}\n{ch.text}")
    out = "\n\n---\n\n".join(parts)
    if max_chars > 0 and len(out) > max_chars:
        out = out[: max_chars - 20] + "\n\n[...truncated...]"
    return out


class S3VectorsRAGRetriever:
    """
    Query Amazon S3 Vector Buckets (boto3 service 's3vectors'), then resolve
    chunk text from local JSONL files keyed by metadata source_file + source_line
    (same convention as rag/aws_vectorDB.py).
    """

    def __init__(
        self,
        *,
        vector_bucket_name: str,
        index_name: str,
        region_name: str,
        kb_dir: str | Path,
        embed_model_name: str | None = None,
        top_k: int = 5,
        max_context_chars: int = 12000,
    ) -> None:
        self.vector_bucket_name = vector_bucket_name
        self.index_name = index_name
        self.region_name = region_name
        self.kb_dir = Path(kb_dir)
        self.embed_model_name = embed_model_name or os.environ.get(
            "EMBED_MODEL", "intfloat/multilingual-e5-small"
        )
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self._client = s3vectors_client(region_name=region_name)
        self._embedder = None

    def _encode_query(self, query: str) -> list[float]:
        # Match rag/full_embeddings.py: passage side uses "passage: "; E5 uses "query: " for queries.
        from sentence_transformers import SentenceTransformer
        import torch

        if self._embedder is None:
            # Default CPU on Modal when sharing a GPU with a 7B LM (set RAG_EMBED_DEVICE=cuda if you prefer).
            device = os.environ.get("RAG_EMBED_DEVICE", "cpu")
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
            self._embedder = SentenceTransformer(
                self.embed_model_name, trust_remote_code=True, device=device
            )
            self._embedder.max_seq_length = int(
                os.environ.get("MAX_SEQ_LENGTH", "512" if device == "cpu" else "1024")
            )

        text = query if query.lower().startswith("query:") else f"query: {query}"
        vec = self._embedder.encode(
            text, convert_to_numpy=True, show_progress_bar=False
        )
        return vec.astype("float32").tolist()

    def retrieve(self, query_en: str) -> tuple[str, list[RetrievedChunk]]:
        qvec = self._encode_query(query_en)
        resp = self._client.query_vectors(
            vectorBucketName=self.vector_bucket_name,
            indexName=self.index_name,
            topK=self.top_k,
            queryVector={"float32": qvec},
            returnMetadata=True,
            returnDistance=True,
        )

        vectors = resp.get("vectors", []) or []
        chunks: list[RetrievedChunk] = []

        for item in vectors:
            key = str(item.get("key", ""))
            distance = item.get("distance")
            if distance is not None:
                try:
                    distance = float(distance)
                except (TypeError, ValueError):
                    distance = None

            meta = item.get("metadata") or {}
            source_file = str(meta.get("source_file", ""))
            try:
                source_line = int(meta.get("source_line", -1))
            except (TypeError, ValueError):
                source_line = -1

            text = ""
            paths = _guess_kb_paths(self.kb_dir, source_file) if source_file else []
            for p in paths:
                rec = _read_jsonl_line(p, source_line)
                if rec:
                    text = _chunk_text_from_record(rec)
                    if text:
                        break

            if not text and key:
                text = f"(no local text resolved for key={key}; source_file={source_file} line={source_line})"

            chunks.append(
                RetrievedChunk(
                    key=key,
                    distance=distance,
                    source_file=source_file,
                    source_line=source_line,
                    text=text,
                )
            )

        ctx = format_context(chunks, max_chars=self.max_context_chars)
        return ctx, chunks
