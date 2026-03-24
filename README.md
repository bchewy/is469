# EN->JA Translation (S0/S1/S2/S3)

Minimal scaffold for:
- S0: Prompt-only baseline
- S1: Fine-tuned adapters
- S2: Fine-tuned + RAG
- S3: Fine-tuned + agentic RAG (LangGraph-style flow)

## Quickstart

1) Install deps

```bash
python -m pip install -e .
```

2) Authenticate Modal

```bash
modal setup
```

3) Create environments

```bash
modal environment create dev
modal environment create prod
modal config set-environment dev
```

4) Create required secrets

```bash
modal secret create enja-hf HF_TOKEN="$HF_TOKEN" --env dev --force
modal secret create enja-s3 \
  AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
  AWS_DEFAULT_REGION="$AWS_DEFAULT_REGION" \
  S3_BUCKET="is469-genai-brianchew" \
  --env dev --force
```

5) Create required volumes

```bash
modal volume create enja-base-models --env dev
modal volume create enja-model-artifacts --env dev
modal volume create enja-rag-index --env dev
modal volume create enja-data --env dev
```

## Qwen bootstrap (HF -> Modal Volume -> S3)

Download `Qwen/Qwen2.5-7B-Instruct` into Modal volume:

```bash
modal run -m modal_jobs.download_qwen --repo-id Qwen/Qwen2.5-7B-Instruct --env dev --timestamps
```

Verify volume contents:

```bash
modal volume ls enja-base-models /models --env dev
```

Sync the downloaded model to S3:

```bash
modal run -m modal_jobs.sync_models_to_s3 \
  --model-path /models/qwen2.5-7b-instruct \
  --s3-prefix models/qwen2.5-7b-instruct \
  --env dev --timestamps
```

Optional one-shot path (download and upload in one job):

```bash
modal run -m modal_jobs.download_qwen_to_s3 --repo-id Qwen/Qwen2.5-7B-Instruct --env dev --timestamps
```
