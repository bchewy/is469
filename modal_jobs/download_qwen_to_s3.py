from __future__ import annotations

from pathlib import Path
import re

import modal

APP_NAME = "enja-download-qwen-to-s3"
TMP_ROOT = Path("/tmp/model-download")

app = modal.App(APP_NAME)
image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "huggingface-hub==0.32.5",
    "boto3==1.37.38",
)


def _repo_to_dirname(repo_id: str) -> str:
    cleaned = repo_id.strip().lower().replace("/", "-")
    return re.sub(r"[^a-z0-9._-]", "-", cleaned)


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("enja-hf", required_keys=["HF_TOKEN"]),
        modal.Secret.from_name(
            "enja-s3",
            required_keys=[
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "AWS_DEFAULT_REGION",
                "S3_BUCKET",
            ],
        ),
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
    import boto3
    from huggingface_hub import snapshot_download

    work_dir = TMP_ROOT / _repo_to_dirname(repo_id)
    work_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(work_dir),
        token=os.environ["HF_TOKEN"],
    )

    bucket = os.environ["S3_BUCKET"]
    client = boto3.client(
        "s3",
        region_name=os.environ["AWS_DEFAULT_REGION"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )

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
