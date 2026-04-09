from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class GlossaryEntry:
    source_term_en: str
    approved_ja: str
    usage_note: str
    forbidden_variants: list[str]


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_glossary",
            "description": (
                "Look up the approved Japanese translation for an English term. "
                "Returns the mandatory approved Japanese form and any forbidden variants. "
                "Use this for any technical, UI, or domain-specific terms before translating."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "term": {
                        "type": "string",
                        "description": "English term to look up (case-insensitive)",
                    }
                },
                "required": ["term"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Semantic search across the translation knowledge base: "
                "translation memory (previous EN-JA translations), style guides, "
                "grammar references, and parallel corpus examples. "
                "Use this to find similar translations, formatting conventions, "
                "or grammar patterns relevant to the source text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query in English or Japanese",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results (default 5, max 20)",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


class ToolExecutor:
    """Dispatches tool calls to glossary lookups and S3 Vectors retrieval."""

    def __init__(
        self,
        *,
        glossary_path: str | Path,
        retriever: Any | None = None,
    ) -> None:
        self.glossary = self._load_glossary(Path(glossary_path))
        self.retriever = retriever
        self.call_log: list[dict] = []

    @property
    def available_tool_names(self) -> list[str]:
        names = ["lookup_glossary"]
        if self.retriever is not None:
            names.append("search_knowledge_base")
        return names

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return OpenAI-format tool definitions for available tools only."""
        available = set(self.available_tool_names)
        return [
            t for t in TOOL_DEFINITIONS
            if t["function"]["name"] in available
        ]

    @staticmethod
    def _load_glossary(path: Path) -> dict[str, GlossaryEntry]:
        entries: dict[str, GlossaryEntry] = {}
        if not path.is_file():
            return entries
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                term = row["source_term_en"].strip().lower()
                forbidden_raw = row.get("forbidden_variants", "") or ""
                forbidden = [
                    v.strip() for v in forbidden_raw.split("|") if v.strip()
                ]
                entries[term] = GlossaryEntry(
                    source_term_en=row["source_term_en"].strip(),
                    approved_ja=row["approved_ja"].strip(),
                    usage_note=(row.get("usage_note") or "").strip(),
                    forbidden_variants=forbidden,
                )
        return entries

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        if name == "lookup_glossary":
            return self._lookup_glossary(arguments["term"])
        if name == "search_knowledge_base":
            return self._search_kb(
                arguments["query"], arguments.get("top_k", 5)
            )
        return json.dumps({"error": f"Unknown tool: {name}"})

    def _lookup_glossary(self, term: str) -> str:
        key = term.strip().lower()

        if key in self.glossary:
            e = self.glossary[key]
            result: dict[str, Any] = {
                "found": True,
                "term": e.source_term_en,
                "approved_ja": e.approved_ja,
                "usage_note": e.usage_note,
            }
            if e.forbidden_variants:
                result["forbidden_variants"] = e.forbidden_variants
            self.call_log.append(
                {"tool": "lookup_glossary", "term": term, "hit": True}
            )
            return json.dumps(result, ensure_ascii=False)

        matches = [
            e
            for k, e in self.glossary.items()
            if key in k or k in key
        ]
        if matches:
            results = [
                {"term": e.source_term_en, "approved_ja": e.approved_ja}
                for e in matches[:5]
            ]
            self.call_log.append(
                {"tool": "lookup_glossary", "term": term, "hit": "partial"}
            )
            return json.dumps(
                {"found": False, "partial_matches": results},
                ensure_ascii=False,
            )

        self.call_log.append(
            {"tool": "lookup_glossary", "term": term, "hit": False}
        )
        return json.dumps(
            {"found": False, "message": f"No glossary entry for '{term}'"}
        )

    def _search_kb(self, query: str, top_k: int = 5) -> str:
        if self.retriever is None:
            self.call_log.append(
                {
                    "tool": "search_knowledge_base",
                    "query": query,
                    "error": "no_retriever",
                }
            )
            return json.dumps(
                {
                    "error": (
                        "Knowledge base search unavailable "
                        "(S3 Vectors not configured)"
                    )
                }
            )

        top_k = min(max(1, int(top_k)), 20)
        saved = self.retriever.top_k
        self.retriever.top_k = top_k
        try:
            _ctx, chunks = self.retriever.retrieve(query)
        finally:
            self.retriever.top_k = saved

        results = [
            {
                "source": c.source_file,
                "text": c.text[:500] if len(c.text) > 500 else c.text,
                "distance": c.distance,
            }
            for c in chunks
        ]
        self.call_log.append(
            {
                "tool": "search_knowledge_base",
                "query": query,
                "count": len(results),
            }
        )
        return json.dumps(
            {"results": results, "count": len(results)}, ensure_ascii=False
        )

    def reset_log(self) -> None:
        self.call_log = []
