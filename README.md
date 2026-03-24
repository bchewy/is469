# EN -> JA Translation Pipeline (S0/S1/S2/S3)

This repo is the early scaffold for an English-to-Japanese SaaS localization pipeline with four planned variants:

- `S0`: prompt-only baseline
- `S1`: fine-tuned adapters
- `S2`: fine-tuned + vanilla RAG
- `S3`: fine-tuned + agentic RAG

Current status: infra/config skeleton is in place, but core model training/inference/retrieval logic is still scaffold-only.

## What Exists Right Now

- Modal job entrypoints under `modal_jobs/`
- Variant/config definitions under `configs/`
- Seed KB files under `kb/`
- Typed schema dataclasses under `src/utils/schemas.py`
- Project packaging in `pyproject.toml`

The following jobs currently execute and return scaffold metadata (not real training/retrieval yet):

- `modal_jobs.run_variant`
- `modal_jobs.train_translation`
- `modal_jobs.train_error_id`
- `modal_jobs.build_index`

The following jobs are implemented for model transfer/bootstrap workflows:

- `modal_jobs.download_qwen` (HF -> Modal volume)
- `modal_jobs.sync_models_to_s3` (Modal volume -> S3)
- `modal_jobs.download_qwen_to_s3` (HF -> local temp -> S3)

## Repository Layout

```text
configs/       Variant and training/retrieval YAML configs
kb/            Seed glossary, translation memory, style guide, grammar notes
modal_jobs/    Modal jobs for scaffold runs and model transfer
src/           Shared Python package (schemas and future modules)
scripts/       Utility scripts placeholder
```

## Requirements

- Python `>=3.11`
- Modal CLI authenticated (`modal setup`)
- Access to:
  - Hugging Face token (`HF_TOKEN`)
  - AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, region, bucket)

Install:

```bash
python -m pip install -e .
```

## Modal Setup

Create/select environment:

```bash
modal environment create dev
modal environment create prod
modal config set-environment dev
```

Create secrets:

```bash
modal secret create enja-hf HF_TOKEN="$HF_TOKEN" --env dev --force
modal secret create enja-s3 \
  AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
  AWS_DEFAULT_REGION="$AWS_DEFAULT_REGION" \
  S3_BUCKET="$S3_BUCKET" \
  --env dev --force
```

Create volumes:

```bash
modal volume create enja-base-models --env dev
modal volume create enja-model-artifacts --env dev
modal volume create enja-rag-index --env dev
modal volume create enja-data --env dev
```

## Run Current Scaffold Jobs

Run variant orchestration stub:

```bash
modal run -m modal_jobs.run_variant --variant s0 --config configs/base.yaml --env dev --timestamps
```

Run fine-tune stubs:

```bash
modal run -m modal_jobs.train_translation --config configs/finetune_translation.yaml --env dev --timestamps
modal run -m modal_jobs.train_error_id --config configs/finetune_error_id.yaml --env dev --timestamps
```

Run retrieval index stub:

```bash
modal run -m modal_jobs.build_index --config configs/rag.yaml --env dev --timestamps
```

## Qwen Model Bootstrap

Download `Qwen/Qwen2.5-7B-Instruct` into Modal volume:

```bash
modal run -m modal_jobs.download_qwen \
  --repo-id Qwen/Qwen2.5-7B-Instruct \
  --env dev --timestamps
```

Verify files in volume:

```bash
modal volume ls enja-base-models /models --env dev
```

Sync downloaded model to S3:

```bash
modal run -m modal_jobs.sync_models_to_s3 \
  --model-path /models/qwen2.5-7b-instruct \
  --s3-prefix models/qwen2.5-7b-instruct \
  --env dev --timestamps
```

One-shot download + upload:

```bash
modal run -m modal_jobs.download_qwen_to_s3 \
  --repo-id Qwen/Qwen2.5-7B-Instruct \
  --s3-prefix models/qwen2.5-7b-instruct \
  --env dev --timestamps
```

## Config Notes

- Base config: `configs/base.yaml`
- Fine-tune configs:
  - `configs/finetune_translation.yaml`
  - `configs/finetune_error_id.yaml`
- Retrieval configs:
  - `configs/rag.yaml`
  - `configs/agentic_rag.yaml`

All configs currently define run parameters and intended behavior, but do not yet correspond to fully implemented pipelines.

## Data and Security Hygiene

- Real credentials belong only in local `.env` (already gitignored).
- `.env.example` contains placeholders only.
- Local model directory `qwen-qwen2.5-7b-instruct/` is gitignored to prevent accidental weight pushes.
- `data/raw/`, `data/processed/`, `data/splits/`, and `results/` are gitignored.

## Next Implementation Milestones

1. Implement S0 inference path and JSON output validation.
2. Implement S1 QLoRA training + adapter artifact persistence.
3. Implement S2 retrieval indexing/querying and trace capture.
4. Implement S3 agentic loop with rewrite/retry controls.
