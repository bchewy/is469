# AGENTS.md

## Cursor Cloud specific instructions

### Overview

This is an EN→JA machine translation pipeline (`genai-llms`). It uses **Modal** for serverless GPU compute (training/inference) and runs data-pipeline scripts locally in Python.

### Quick reference

- **Install**: `pip install -e .` (editable install from `pyproject.toml`)
- **Tests**: `python3 -m pytest tests/ -v` (12 unit tests, all local, no GPU/cloud needed)
- **Lint**: `ruff check .` (not configured in `pyproject.toml` but works out of the box; 11 pre-existing warnings)
- **Local data pipeline demo**: `python3 -m scripts.build_kb_splits --train-count 10 --dev-count 5 --test-count 5`
- **Build glossary**: `python3 -m scripts.build_glossary --input data/splits/train_v1.jsonl --output results/glossary.csv`
- **Build translation memory**: `python3 -m scripts.build_translation_memory --input data/splits/train_v1.jsonl --output results/tm.jsonl`

### Key caveats

- `python` is not on PATH in this environment; always use `python3`.
- Scripts installed to `~/.local/bin` (e.g. `modal`, `ruff`, `sacrebleu`) need `PATH="$HOME/.local/bin:$PATH"`.
- All GPU workloads (training, inference) run on **Modal** and require `modal setup` + secrets (`enja-hf`, `enja-s3-models`, `enja-s3-vectors`). They cannot run locally without a GPU. See `README.md` for full Modal setup.
- `pytest` is not a declared project dependency; install it separately if needed.
- No Docker or docker-compose is used; Modal handles containerization.
- The `data/`, `results/`, and model directories are gitignored.
