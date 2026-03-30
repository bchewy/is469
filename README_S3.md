# S3 Agentic RAG Handoff

This document is the focused handoff guide for the `S3` path in this repo:

- `fine-tuned + agentic RAG`
- retrieval from Amazon S3 Vectors
- translation + self-audit + evaluation

Use this file instead of the main `README.md` when the goal is only to run, debug, or extend the `S3` pipeline.

## What S3 Currently Does

The `S3` path runs from [modal_jobs/run_s3.py](modal_jobs/run_s3.py) with config [configs/s3_inference.yaml](configs/s3_inference.yaml).

At a high level it:

1. loads the base Qwen model
2. loads the fine-tuned LoRA adapter if present
3. retrieves context from Amazon S3 Vectors
4. generates a Japanese translation
5. runs a self-audit / rewrite loop
6. runs a structured error-check pass
7. writes predictions and metrics

## Architecture Diagram

```mermaid
flowchart TD
    A[Test JSONL<br/>data/splits/test_v1.jsonl] --> B[modal_jobs/run_s3.py]
    B --> C[Load config<br/>configs/s3_inference.yaml]
    C --> D[Load base model<br/>/models or HF]
    C --> E[Load LoRA adapter<br/>/artifacts/adapters/translation/final]
    C --> F[Init S3 Vectors retriever]

    A --> G[Per example]
    G --> H[Retrieve top-k context]
    H --> I[Agentic translation loop]
    I --> J[Initial translation]
    J --> K[Critic/self-audit]
    K -->|coverage low| L[Rewrite / revise]
    L --> K
    K -->|coverage ok| M[Final translation]

    M --> N[Error-check JSON]
    H --> O[Retrieval eval]
    M --> P[Terminology eval]
    M --> Q[Translation metrics]
    N --> R[Error-ID metrics]

    Q --> S[s3_metrics.json]
    R --> S
    O --> S
    P --> S
    M --> T[s3_outputs.jsonl]
```

## Important Files

- Entry point: [modal_jobs/run_s3.py](modal_jobs/run_s3.py)
- Agent loop: [src/agents/agentic_rag.py](src/agents/agentic_rag.py)
- Retrieval: [src/retrieval/s3_vectors_rag.py](src/retrieval/s3_vectors_rag.py)
- Evaluation helpers: [src/eval/s3_eval.py](src/eval/s3_eval.py)
- Prompts: [src/prompts/s3_prompts.py](src/prompts/s3_prompts.py)
- S3 config: [configs/s3_inference.yaml](configs/s3_inference.yaml)
- Data split generator:
  - simple: [scripts/build_kb_splits.py](scripts/build_kb_splits.py)
  - more realistic: [scripts/build_realistic_kb_splits.py](scripts/build_realistic_kb_splits.py)
- KB assets:
  - [kb/glossary.csv](kb/glossary.csv)
  - [kb/translation_memory.jsonl](kb/translation_memory.jsonl)
  - [kb/annotations_raw.jsonl](kb/annotations_raw.jsonl)
  - [kb/gemini_annotated_results.jsonl](kb/gemini_annotated_results.jsonl)
  - [kb/eng-jap.tsv](kb/eng-jap.tsv)

## Current Inputs and Outputs

### Input

The current config expects:

- `data/splits/test_v1.jsonl`

Each row should look like:

```json
{
  "id": "annot-12345_67890",
  "source_en": "Please update your password.",
  "target_ja": "パスワードを更新してください。",
  "domain": "general",
  "source_ref": "annotations_raw.jsonl",
  "quality_score": 0.95,
  "license": "unknown",
  "split": "test",
  "group_key": "please update your password."
}
```

### Output

`run_s3` writes:

- `results/metrics/s3_outputs.jsonl`
- `results/metrics/s3_metrics.json`

These live in the `enja-results` Modal volume.

## Prerequisites

You need:

- Modal CLI authenticated
- `HF_TOKEN`
- `VECTORS_AWS_ACCESS_KEY_ID`
- `VECTORS_AWS_SECRET_ACCESS_KEY`
- `VECTORS_AWS_DEFAULT_REGION`

PowerShell setup:

```powershell
.\venv\Scripts\python.exe -m pip install -e .
.\venv\Scripts\modal.exe secret create enja-hf --from-dotenv .env --env dev --force
.\venv\Scripts\modal.exe secret create enja-s3-vectors --from-dotenv .env --env dev --force
```

Create volumes if needed:

```powershell
.\venv\Scripts\modal.exe volume create enja-base-models --env dev
.\venv\Scripts\modal.exe volume create enja-model-artifacts --env dev
.\venv\Scripts\modal.exe volume create enja-data --env dev
.\venv\Scripts\modal.exe volume create enja-results --env dev
```

## Recommended Run Sequence

### 1. Generate realistic train/dev/test splits

```powershell
.\venv\Scripts\python.exe -m scripts.build_realistic_kb_splits --train-count 2000 --dev-count 250 --test-count 250
```

This creates:

- `data/splits/train_v1.jsonl`
- `data/splits/dev_v1.jsonl`
- `data/splits/test_v1.jsonl`

### 2. Upload splits to Modal

```powershell
.\venv\Scripts\modal.exe volume put enja-data .\data\splits\train_v1.jsonl /data/splits/train_v1.jsonl --env dev --force
.\venv\Scripts\modal.exe volume put enja-data .\data\splits\dev_v1.jsonl /data/splits/dev_v1.jsonl --env dev --force
.\venv\Scripts\modal.exe volume put enja-data .\data\splits\test_v1.jsonl /data/splits/test_v1.jsonl --env dev --force
```

### 3. Make sure the base model is available

Option A: store it in Modal volume:

```powershell
.\venv\Scripts\modal.exe run -m modal_jobs.download_qwen --repo-id Qwen/Qwen2.5-7B-Instruct --env dev --timestamps
```

Option B: let `run_s3` pull it from Hugging Face using `HF_TOKEN`.

### 4. Train the translation adapter

```powershell
.\venv\Scripts\modal.exe run -m modal_jobs.train_translation --config configs/finetune_translation.yaml --env dev --timestamps
```

Check that the adapter exists:

```powershell
.\venv\Scripts\modal.exe volume ls enja-model-artifacts /adapters/translation/final --env dev
```

### 5. Run S3 inference

```powershell
.\venv\Scripts\modal.exe run -m modal_jobs.run_s3 --config configs/s3_inference.yaml --env dev --timestamps
```

### 6. Fetch outputs and metrics

```powershell
.\venv\Scripts\modal.exe volume get enja-results /results/metrics/s3_metrics.json .\results\ --env dev --force
.\venv\Scripts\modal.exe volume get enja-results /results/metrics/s3_outputs.jsonl .\results\ --env dev --force
```

## Current S3 Metrics Implemented

The current `S3` run produces:

- Translation quality:
  - `BLEU`
  - `chrF++`
  - `COMET`
- System:
  - `avg_latency_ms`
  - `avg_retrieval_ms`
- Agent loop:
  - `avg_coverage_score`
  - `total_rewrite_steps`
  - `total_revision_steps`
- Retrieval:
  - `retrieval_hit_at_k`
  - `retrieval_recall_at_k`
- Terminology:
  - `terminology_accuracy`
- Error identification:
  - `error_binary_f1`
  - `error_category_macro_f1`

Per-example outputs also include:

- `retrieval_chunks`
- `retrieval_eval`
- `terminology_eval`
- `error_check`
- `gold_error_label`
- `agent_trace`

## What Is Working Well

As of the latest runs:

- the full `S3` path runs end to end
- translation metrics are now meaningful
- `COMET`, `BLEU`, and `chrF++` are being computed
- terminology evaluation now has non-zero coverage
- retrieval timing is being logged
- the prompt-based error-check pass is running

## Remaining Work / Known Issues

These are the main things the team still needs to finish.

### 1. Fix error-ID gold label lookup

Current symptom:

- `error_id_eval_samples` is still `0`

Most likely cause:

- in [modal_jobs/run_s3.py](modal_jobs/run_s3.py), `gold_error_label` is still being looked up with the raw row ID:

```python
gold_error_label = eval_assets.gold_error_by_id.get(row_id)
```

But the test split currently uses IDs like `annot-...`, while the gold labels in [kb/gemini_annotated_results.jsonl](kb/gemini_annotated_results.jsonl) use the unprefixed ID.

Expected fix:

- canonicalize `row_id` before lookup, using the same normalization logic used in [src/eval/s3_eval.py](src/eval/s3_eval.py)

Why this matters:

- until this is fixed, `error_binary_f1` and `error_category_macro_f1` are not trustworthy because they are not being evaluated on any rows

### 2. Make retrieval evaluation more faithful to the actual index

Current symptom:

- `retrieval_hit_at_k` and `retrieval_recall_at_k` are still `0.0`

Likely cause:

- the current retrieval metric is still a proxy / heuristic
- the vector index appears to be built mostly from chunked corpora, not a clean glossary/TM-only target set
- exact evidence matching is difficult with chunked retrieval

Recommended next step:

- instrument the retrieval index build so each query has a clearer gold retrieval target
- or add retrieval evaluation for:
  - glossary term hit
  - translation memory example hit
  - exact chunk/source-line support

Why this matters:

- right now translation quality can improve even if retrieval metrics remain `0.0`, which makes it hard to prove that RAG itself is helping

### 3. Improve glossary coverage

Current state:

- glossary coverage is better than before, but still limited

Recommended next step:

- expand [kb/glossary.csv](kb/glossary.csv) with more domain-relevant, high-value terms
- prefer stable terminology where one approved Japanese form should win over alternatives
- make sure the dev/test split contains a meaningful glossary-sensitive subset

Why this matters:

- terminology accuracy is one of the cleanest ways to show the value of RAG in this project

### 4. Add retrieval corpus cleanup / audit

Recommended next step:

- confirm exactly which sources are in the S3 vector index
- verify whether glossary and translation memory are actually indexed, not just present in `kb/`
- document the current indexed sources in this handoff file

Why this matters:

- the kickoff plan assumes glossary + translation memory + style guide + grammar notes are active RAG assets
- if the index does not contain them, retrieval metrics and downstream behavior will not match the intended design

### 5. Create a qualitative review set

Recommended next step:

- manually inspect 20–30 examples from `s3_outputs.jsonl`
- group them by:
  - terminology wins
  - retrieval failures
  - fluency issues
  - major mistranslations
  - cases where rewrite/revision helped

Why this matters:

- this is required by the kickoff doc
- it will also help explain whether agentic RAG is actually adding value over S2

### 6. Run the actual comparison table

Needed comparison chain:

- `S0 -> S1`
- `S1 -> S2`
- `S2 -> S3`

Current status:

- `S3` is the most actively worked path
- the full comparison table is not yet documented in one place

Recommended next step:

- run each variant on the same held-out test set
- store outputs side by side
- summarize in a single experiment table

## Quick Troubleshooting

### Adapter not loaded

If you see:

- `adapter_dir not found: /artifacts/adapters/translation/final`

Then either:

- the adapter has not been trained yet
- or it was not uploaded into the `enja-model-artifacts` Modal volume

### Input file not found

Remember that the code resolves:

- `data/splits/test_v1.jsonl`

inside the mounted data volume, so upload paths must match what `run_s3.py` expects.

### Retrieval is enabled but metrics stay zero

That does not automatically mean retrieval is broken. It may mean:

- the retrieval metric is too strict
- the indexed sources do not match the eval assumptions
- the current test examples are not ideal retrieval targets

### BLEU/chrF/COMET disagree

This is normal.

- `COMET` is the strongest MT metric here
- `BLEU` is still useful but harsh
- `chrF++` is often more forgiving for Japanese

Use all three together.

## Handoff Summary

If the next teammate only has 10 minutes, tell them this:

1. `modal_jobs/run_s3.py` is the entry point.
2. `configs/s3_inference.yaml` controls the run.
3. `data/splits/test_v1.jsonl` is the current eval input.
4. `results/metrics/s3_metrics.json` and `s3_outputs.jsonl` are the main outputs.
5. Translation metrics are working.
6. Terminology evaluation is partially working.
7. Retrieval evaluation still needs better grounding.
8. Error-ID F1 still needs the ID lookup fix in `run_s3.py`.
