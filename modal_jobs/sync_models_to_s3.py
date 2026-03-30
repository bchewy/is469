from __future__ import annotations

import sys
from pathlib import Path

import modal

APP_NAME = "enja-sync-models-to-s3"
MODELS_DIR = Path("/models")
ROOT = Path(__file__).resolve().parents[1]

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("boto3==1.37.38")
    .add_local_dir(ROOT / "src", remote_path="/root/src")
)
models_volume = modal.Volume.from_name("enja-base-models", create_if_missing=True)


@app.function(
    image=image,
    volumes={str(MODELS_DIR): models_volume},
    # Object S3 (fine-tuned weights / model sync) — account A. Use MODELS_* keys.
    secrets=[modal.Secret.from_name("enja-s3-models")],
    timeout=60 * 60,
)
def sync_model_dir(model_path: str, s3_prefix: str = "models/qwen2.5-7b-instruct") -> int:
    sys.path.insert(0, "/root")
    from src.utils.aws_profiles import boto3_session_for_models, models_s3_bucket

    source_dir = Path(model_path)
    if not source_dir.exists():
        raise FileNotFoundError(f"Model path does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise ValueError(f"Model path must be a directory: {source_dir}")

    bucket = models_s3_bucket()
    session = boto3_session_for_models()
    client = session.client("s3")

    uploaded = 0
    for file_path in source_dir.rglob("*"):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(source_dir).as_posix()
        key = f"{s3_prefix.rstrip('/')}/{rel}"
        client.upload_file(str(file_path), bucket, key)
        uploaded += 1

    print(f"Uploaded {uploaded} files from {source_dir} to s3://{bucket}/{s3_prefix}/")
    return uploaded


@app.local_entrypoint()
def main(
    model_path: str = "/models/qwen2.5-7b-instruct",
    s3_prefix: str = "models/qwen2.5-7b-instruct",
):
    print(sync_model_dir.remote(model_path=model_path, s3_prefix=s3_prefix))
