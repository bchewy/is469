from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
