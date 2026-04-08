# AGENTS.md

## Active Scope

This repository is currently being prepared for a reproducible `S0` vs `S1` rerun on a standard CUDA GPU machine such as Prime Intellect. Unless the user explicitly redirects the work, treat that as the active priority.

## Canonical References

- `S0_S1_GPU_RERUN_AGENT.md` is the concrete rerun runbook.
- `REPORT.md` is the canonical S0/S1 report section.
- `REPORT_S0_S1.md` is the older supporting version of the same report slice.
- `scripts/train_local.py` is the local QLoRA training entrypoint.
- `modal_jobs/run_variant.py` is reference inference/evaluation logic to mirror locally, not the target runtime.

## Report Framing

- Keep the writeup focused on English-to-Japanese translation and localization quality.
- Do not turn the report back into a full-team omnibus report unless the user explicitly asks.
- Do not reframe the project as a language-learning tutor story.

## Locked Experiment Contract

- `S0` means prompt-only `Qwen/Qwen2.5-7B-Instruct` inference.
- `S1` means the same base model plus the QLoRA adapter trained from this repo.
- Training split: `data/splits/train_v2.jsonl`
- Dev split: `data/splits/dev_v2.jsonl`
- Evaluation split: `data/splits/test_v1.jsonl`
- Run both variants on the same non-empty evaluation split.
- Use one glossary snapshot for both variants and save it to `results/metrics/glossary_used_for_s0_s1.csv`.
- Retrieval metrics and error-F1 metrics do not apply to S0/S1. In tables they should remain `—` / `N/A`, never fake zeros.

## Allowed Changes

- Minimal config edits needed to run on a local or Prime Intellect GPU box.
- A local inference runner such as `scripts/run_variant_local.py` that mirrors `modal_jobs/run_variant.py`.
- A metrics-completeness patch that adds `terminology_samples`, `glossary_entries_used`, and optionally `glossary_snapshot_path`.
- Report updates that are driven by saved rerun artifacts.

## Avoid

- Do not change the task definition or prompt setup just to chase better numbers.
- Do not switch the rerun to `test_v2.jsonl`; it is empty and would change the experiment.
- Do not mix evaluation sets, sample sizes, or glossary snapshots across `S0` and `S1`.
- Do not add retrieval, RAG, agentic loops, or extra evaluation logic to the plain S0/S1 rerun path.
- Do not hide flat results. A mostly flat `S1` result is acceptable if it is honest and reproducible.
- Do not attempt 7B QLoRA retraining on this VPS if no usable GPU is available; move to a real CUDA machine instead.

## Prime Intellect / Local GPU Workflow

- Prefer a normal CUDA Linux workflow, not Modal.
- Prefer repo-local paths like `models/...` and `results/metrics/...` over provider-specific mount conventions when practical.
- Treat Modal files as reference implementations only.
- If you need a bootstrap helper for a remote GPU box, check `scripts/setup_and_train.sh`, but do not mistake it for the full rerun procedure.

## Required Artifacts

- `results/metrics/s0_metrics.json`
- `results/metrics/s1_metrics.json`
- `results/metrics/s0_outputs.jsonl`
- `results/metrics/s1_outputs.jsonl`
- `results/metrics/glossary_used_for_s0_s1.csv`
- updated `REPORT.md` and/or `REPORT_S0_S1.md`
- if training is rerun: the final adapter directory and `train_results.json`

## Verification

- Run `python -m unittest discover -s tests -v` before and after non-trivial code changes when the environment is ready.
- Confirm the evaluation split is non-empty before long GPU runs.
- Confirm the saved metrics explicitly include terminology sample counts after the rerun patch.
- Ensure the report states whether `S1` improved, worsened, or remained flat versus `S0`.
