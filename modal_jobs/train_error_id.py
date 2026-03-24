from __future__ import annotations

import modal

from modal_jobs.common import load_yaml

app = modal.App("enja-train-error-id")
image = modal.Image.debian_slim(python_version="3.11").uv_pip_install("pyyaml==6.0.2")


@app.function(image=image, timeout=60 * 60)
def train(config: str) -> dict:
    cfg = load_yaml(config)
    result = {
        "status": "scaffold_only",
        "job": "train_error_id",
        "config": config,
        "training": cfg.get("training", {}),
    }
    print(result)
    return result


@app.local_entrypoint()
def main(config: str = "configs/finetune_error_id.yaml"):
    print(train.remote(config=config))
