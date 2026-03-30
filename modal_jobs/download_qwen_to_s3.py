from __future__ import annotations

import sys
from pathlib import Path
import re

import modal

APP_NAME = "enja-download-qwen-to-s3"
TMP_ROOT = Path("/tmp/model-download")
ROOT = Path(__file__).resolve().parents[1]

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "huggingface-hub==0.32.5",
        "boto3==1.37.38",
    )
    .add_local_dir(ROOT / "src", remote_path="/root/src")
)


def _repo_to_dirname(repo_id: str) -> str:
    cleaned = repo_id.strip().lower().replace("/", "-")
    return re.sub(r"[^a-z0-9._-]", "-", cleaned)


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("enja-hf", required_keys=["HF_TOKEN"]),
        modal.Secret.from_name("enja-s3-models"),
    ],
    timeout=60 * 60 * 3,
    ephemeral_disk=60 * 1024,
)
def download_and_upload(
    repo_id: str = "Qwen/Qwen2.5-7B-Instruct",
    revision: str | None = None,
    s3_prefix: str = "models/qwen2.5-7b-instruct",
) -> int:
    import os
    from huggingface_hub import snapshot_download

    sys.path.insert(0, "/root")
    from src.utils.aws_profiles import boto3_session_for_models, models_s3_bucket

    work_dir = TMP_ROOT / _repo_to_dirname(repo_id)
    work_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(work_dir),
        token=os.environ["HF_TOKEN"],
    )

    bucket = models_s3_bucket()
    client = boto3_session_for_models().client("s3")

    uploaded = 0
    for file_path in work_dir.rglob("*"):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(work_dir).as_posix()
        key = f"{s3_prefix.rstrip('/')}/{rel}"
        client.upload_file(str(file_path), bucket, key)
        uploaded += 1

    print(f"Uploaded {uploaded} files to s3://{bucket}/{s3_prefix}/")
    return uploaded


@app.local_entrypoint()
def main(
    repo_id: str = "Qwen/Qwen2.5-7B-Instruct",
    revision: str = "",
    s3_prefix: str = "models/qwen2.5-7b-instruct",
):
    revision_value = revision if revision else None
    print(
        download_and_upload.remote(
            repo_id=repo_id, revision=revision_value, s3_prefix=s3_prefix
        )
    )
