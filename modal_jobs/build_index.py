from __future__ import annotations

import modal

from modal_jobs.common import load_yaml

app = modal.App("enja-build-index")
image = modal.Image.debian_slim(python_version="3.11").uv_pip_install("pyyaml==6.0.2")


@app.function(image=image, timeout=60 * 20)
def build(config: str) -> dict:
    cfg = load_yaml(config)
    retrieval_cfg = cfg.get("retrieval", {})
    result = {
        "status": "scaffold_only",
        "config": config,
        "top_k": retrieval_cfg.get("top_k"),
        "kb_paths": retrieval_cfg.get("kb_paths", []),
    }
    print(result)
    return result


@app.local_entrypoint()
def main(config: str = "configs/rag.yaml"):
    print(build.remote(config=config))
