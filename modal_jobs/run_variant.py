from __future__ import annotations

import modal

from modal_jobs.common import load_yaml

app = modal.App("enja-run-variant")
image = modal.Image.debian_slim(python_version="3.11").uv_pip_install("pyyaml==6.0.2")


@app.function(image=image, timeout=60 * 20)
def run(variant: str, config: str) -> dict:
    cfg = load_yaml(config)
    result = {
        "status": "scaffold_only",
        "variant": variant,
        "config": config,
        "config_keys": sorted(cfg.keys()),
    }
    print(result)
    return result


@app.local_entrypoint()
def main(variant: str = "s0", config: str = "configs/base.yaml"):
    print(run.remote(variant=variant, config=config))
