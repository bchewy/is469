from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


VALID_SPLITS = ("train", "dev", "test")
VALID_DOMAINS = (
    "general",
    "software_docs",
    "customer_support",
    "marketing",
    "ui_strings",
    "legal",
)
VALID_LICENSES = (
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "cc-by-nc-4.0",
    "cc0-1.0",
    "mit",
    "apache-2.0",
    "custom",
    "unknown",
)


# ---------------------------------------------------------------------------
# Dataset row schema
# ---------------------------------------------------------------------------
@dataclass
class TranslationRow:
    id: str
    source_en: str
    target_ja: str
    domain: str = "general"
    source_ref: str = ""
    quality_score: float = 1.0
    license: str = "unknown"
    split: str = ""
    group_key: str = ""

    @staticmethod
    def generate_id(prefix: str = "row") -> str:
        return f"{prefix}-{uuid.uuid4().hex[:12]}"

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.id:
            errors.append("id is empty")
        if not self.source_en or not self.source_en.strip():
            errors.append("source_en is empty")
        if not self.target_ja or not self.target_ja.strip():
            errors.append("target_ja is empty")
        if self.domain not in VALID_DOMAINS:
            errors.append(f"domain '{self.domain}' not in {VALID_DOMAINS}")
        if self.split and self.split not in VALID_SPLITS:
            errors.append(f"split '{self.split}' not in {VALID_SPLITS}")
        if not (0.0 <= self.quality_score <= 1.0):
            errors.append(f"quality_score {self.quality_score} not in [0,1]")
        if self.license not in VALID_LICENSES:
            errors.append(f"license '{self.license}' not in {VALID_LICENSES}")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TranslationRow:
        d = dict(d)
        if "reference_ja" in d and "target_ja" not in d:
            d["target_ja"] = d.pop("reference_ja")
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        filtered.setdefault("source_en", "")
        filtered.setdefault("target_ja", "")
        filtered.setdefault("id", "")
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Inference / output schemas (kept from original scaffold)
# ---------------------------------------------------------------------------
@dataclass
class TranslationInput:
    source_en: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorJson:
    has_error: bool
    severity: str
    categories: list[str]
    rationale: str


@dataclass
class TranslationOutput:
    translation_ja: str
    error_json: ErrorJson
    retrieval_trace: dict[str, Any] = field(default_factory=dict)
    latency_ms: int = 0
    variant_id: str = "s0"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_jsonl_file(
    path: str | Path,
    *,
    require_split: bool = False,
    expected_split: str | None = None,
) -> tuple[int, list[str]]:
    """Validate every line in a JSONL file against TranslationRow schema.

    Returns (valid_count, list_of_error_strings).
    """
    path = Path(path)
    if not path.exists():
        return 0, [f"File not found: {path}"]

    errors: list[str] = []
    valid = 0

    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"L{lineno}: invalid JSON – {exc}")
                continue

            row = TranslationRow.from_dict(obj)
            row_errors = row.validate()

            if require_split and not row.split:
                row_errors.append("split is required but empty")
            if expected_split and row.split != expected_split:
                row_errors.append(
                    f"expected split='{expected_split}', got '{row.split}'"
                )

            if row_errors:
                errors.append(f"L{lineno} id={row.id}: {'; '.join(row_errors)}")
            else:
                valid += 1

    return valid, errors


def load_rows(path: str | Path) -> list[TranslationRow]:
    """Load all rows from a JSONL file."""
    path = Path(path)
    rows: list[TranslationRow] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(TranslationRow.from_dict(json.loads(line)))
    return rows


def write_rows(rows: list[TranslationRow], path: str | Path) -> int:
    """Write rows to a JSONL file. Returns count written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(row.to_json() + "\n")
    return len(rows)
