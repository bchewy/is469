from __future__ import annotations

from pathlib import Path
import re

import modal

APP_NAME = "enja-download-qwen"
MODELS_DIR = Path("/models")

app = modal.App(APP_NAME)
image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "huggingface-hub==0.32.5"
)
models_volume = modal.Volume.from_name("enja-base-models", create_if_missing=True)


def _repo_to_dirname(repo_id: str) -> str:
    cleaned = repo_id.strip().lower().replace("/", "-")
    return re.sub(r"[^a-z0-9._-]", "-", cleaned)


@app.function(
    image=image,
    volumes={str(MODELS_DIR): models_volume},
    secrets=[modal.Secret.from_name("enja-hf", required_keys=["HF_TOKEN"])],
    timeout=60 * 60 * 2,
    gpu=None,
)
def download_model(repo_id: str, revision: str | None = None) -> str:
    import os
    from huggingface_hub import snapshot_download

    target_dir = MODELS_DIR / _repo_to_dirname(repo_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(target_dir),
        token=os.environ["HF_TOKEN"],
    )
    print(f"Downloaded {repo_id} to {target_dir}")
    return str(target_dir)


@app.local_entrypoint()
def main(
    repo_id: str = "Qwen/Qwen2.5-7B-Instruct",
    revision: str = "",
):
    revision_value = revision if revision else None
    print(download_model.remote(repo_id=repo_id, revision=revision_value))
