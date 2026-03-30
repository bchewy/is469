"""
Split AWS credentials for two accounts:

- **Models / object storage** (S3 PutObject for weights): `MODELS_*` env vars.
- **Vector index** (S3 Vectors API `s3vectors`): `VECTORS_*` env vars.

Legacy single-account: if `MODELS_*` is unset, model jobs fall back to `AWS_*` / `S3_BUCKET`.
Vector retrieval requires `VECTORS_*` when using a separate account (no fallback to `AWS_*`
so the wrong account is never used by mistake).
"""

from __future__ import annotations

import os
from typing import Any

import boto3


def _require(key: str) -> str:
    v = os.environ.get(key, "").strip()
    if not v:
        raise KeyError(key)
    return v


def boto3_session_for_models() -> boto3.session.Session:
    """Session for classic S3 object storage (fine-tuned model uploads, etc.)."""
    if os.environ.get("MODELS_AWS_ACCESS_KEY_ID"):
        return boto3.Session(
            aws_access_key_id=_require("MODELS_AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=_require("MODELS_AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get(
                "MODELS_AWS_DEFAULT_REGION", "ap-southeast-1"
            ),
        )
    # Legacy: one secret with AWS_ACCESS_KEY_ID + S3_BUCKET
    return boto3.Session(
        aws_access_key_id=_require("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=_require("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "ap-southeast-1"),
    )


def models_s3_bucket() -> str:
    return os.environ.get("MODELS_S3_BUCKET") or _require("S3_BUCKET")


def boto3_session_for_vectors(region_name: str | None = None) -> boto3.session.Session:
    """Session for S3 Vectors (`boto3.client('s3vectors', ...)`)."""
    region = region_name or os.environ.get(
        "VECTORS_AWS_DEFAULT_REGION", "ap-southeast-1"
    )
    return boto3.Session(
        aws_access_key_id=_require("VECTORS_AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=_require("VECTORS_AWS_SECRET_ACCESS_KEY"),
        region_name=region,
    )


def s3vectors_client(*, region_name: str | None = None) -> Any:
    """S3 Vectors client: uses `VECTORS_*` when set; else default credential chain (dev only)."""
    r = region_name or os.environ.get("VECTORS_AWS_DEFAULT_REGION", "ap-southeast-1")
    if os.environ.get("VECTORS_AWS_ACCESS_KEY_ID"):
        return boto3_session_for_vectors(region_name=region_name).client(
            "s3vectors", region_name=r
        )
    # Local dev: e.g. `aws configure` or legacy single-account `AWS_*` for the vector account.
    return boto3.client("s3vectors", region_name=r)
