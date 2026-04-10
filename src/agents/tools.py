from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class GlossaryEntry:
    source_term_en: str
    approved_ja: str
    usage_note: str
    forbidden_variants: list[str]


# ---------------------------------------------------------------------------
# Locale validation rules
# ---------------------------------------------------------------------------

# Matches MM/DD or MM/DD/YYYY (not inside URLs or decimals)
_DATE_SLASH_RE = re.compile(r"(?<![/\w])\d{1,2}/\d{1,2}(?:/\d{2,4})?(?![/\w])")
# Matches YYYY-MM-DD ISO dates
_DATE_ISO_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
# Japanese character followed by ASCII period (not decimal, not ellipsis)
_JP_THEN_PERIOD_RE = re.compile(r"[\u3040-\u9FFF\uFF00-\uFFEF][.](?![.])")
# Japanese character followed by ASCII comma
_JP_THEN_COMMA_RE = re.compile(r"[\u3040-\u9FFF\uFF00-\uFFEF],")
# Western double quotes wrapping Japanese content
_WESTERN_QUOTE_RE = re.compile(r'"[^"]*[\u3040-\u9FFF][^"]*"')


def _validate_locale_text(text: str) -> str:
    """Run rule-based Japanese locale checks. Returns JSON result string."""
    issues: list[dict] = []

    date_slashes = _DATE_SLASH_RE.findall(text)
    if date_slashes:
        issues.append({
            "rule": "date_slash_format",
            "severity": "major",
            "matches": date_slashes[:5],
            "suggestion": "Use MM月DD日 (e.g. 3月15日) or YYYY年MM月DD日",
        })

    iso_dates = _DATE_ISO_RE.findall(text)
    if iso_dates:
        issues.append({
            "rule": "date_iso_format",
            "severity": "major",
            "matches": iso_dates[:5],
            "suggestion": "Use YYYY年MM月DD日 (e.g. 2024年3月15日)",
        })

    jp_periods = _JP_THEN_PERIOD_RE.findall(text)
    if jp_periods:
        issues.append({
            "rule": "ascii_period_after_japanese",
            "severity": "major",
            "matches": jp_periods[:5],
            "suggestion": "Use 。(fullwidth period) to end Japanese sentences",
        })

    jp_commas = _JP_THEN_COMMA_RE.findall(text)
    if jp_commas:
        issues.append({
            "rule": "ascii_comma_after_japanese",
            "severity": "minor",
            "matches": jp_commas[:5],
            "suggestion": "Use 、(Japanese comma) between Japanese phrases",
        })

    western_quotes = _WESTERN_QUOTE_RE.findall(text)
    if western_quotes:
        issues.append({
            "rule": "western_quotes_around_japanese",
            "severity": "minor",
            "matches": [q[:40] for q in western_quotes[:3]],
            "suggestion": 'Use 「...」 instead of "..." for Japanese quotations',
        })

    stripped = text.rstrip()
    if stripped.endswith(".") and not stripped.endswith("..."):
        issues.append({
            "rule": "trailing_ascii_period",
            "severity": "major",
            "suggestion": "Translation ends with ASCII period; use 。",
        })

    if not issues:
        return json.dumps(
            {"valid": True, "issues": [], "message": "No locale violations found."},
            ensure_ascii=False,
        )
    return json.dumps(
        {"valid": False, "issue_count": len(issues), "issues": issues},
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Translation memory helpers
# ---------------------------------------------------------------------------

_TM_CACHE: dict[str, list[dict]] = {}
_GRAMMAR_CACHE: dict[str, list[dict]] = {}
_TOKEN_RE = re.compile(r"[a-z']+|\d+")


def _load_tm(path: Path) -> list[dict]:
    key = str(path.resolve())
    if key not in _TM_CACHE:
        entries: list[dict] = []
        if path.is_file():
            with path.open(encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        _TM_CACHE[key] = entries
    return _TM_CACHE[key]


def _load_grammar(path: Path) -> list[dict]:
    key = str(path.resolve())
    if key not in _GRAMMAR_CACHE:
        entries: list[dict] = []
        if path.is_file():
            with path.open(encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        _GRAMMAR_CACHE[key] = entries
    return _GRAMMAR_CACHE[key]


def _jaccard(a: str, b: str) -> float:
    ta = set(_TOKEN_RE.findall(a.lower()))
    tb = set(_TOKEN_RE.findall(b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _grammar_score(query: str, chunk: dict) -> float:
    """Score a grammar chunk against a query.
    Section title matches are weighted 2× over body text matches."""
    section_score = _jaccard(query, chunk.get("section", ""))
    text_score = _jaccard(query, chunk.get("text", "")[:300])
    return 2.0 * section_score + text_score


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

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
    {
        "type": "function",
        "function": {
            "name": "lookup_translation_memory",
            "description": (
                "Find previously translated sentences similar to the source text. "
                "Returns close matches with their approved Japanese translations. "
                "Use this before translating to anchor your output to validated "
                "reference translations, especially for templated or repetitive sentences."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sentence": {
                        "type": "string",
                        "description": "Source English sentence to find TM matches for",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of closest matches to return (default 3, max 10)",
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Minimum similarity score 0–1 (default 0.3)",
                    },
                },
                "required": ["sentence"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_grammar_pattern",
            "description": (
                "Search the Japanese grammar reference for rules, patterns, and examples "
                "relevant to a specific grammar question. Covers particles (は/も/が/を/に/で), "
                "verb conjugation, polite forms (ます/です), conditionals, transitivity, "
                "negation, and more. Use this when you are unsure about a grammatical "
                "construction or particle choice in your translation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Grammar concept or question in plain English "
                            "(e.g. 'は vs が particle', 'polite verb form ます', "
                            "'conditional if-then', 'transitive intransitive')"
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of matching sections to return (default 3, max 8)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_locale",
            "description": (
                "Check a Japanese translation for locale and formatting errors. "
                "Detects wrong date formats (MM/DD → MM月DD日), ASCII punctuation "
                "used instead of Japanese equivalents (. → 。, , → 、), and western "
                'quote styles (" " → 「」). '
                "Call this after producing a draft translation to catch formatting "
                "issues before the critic scores it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Japanese translation text to validate",
                    }
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web to verify translation choices, find real-world "
                "Japanese usage examples, or check cultural context. Use this when "
                "the glossary doesn't have a term, or when you want to verify that "
                "your translation sounds natural in real Japanese usage. "
                "Powered by Brave Search API."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search query — use Japanese for finding usage examples, "
                            "English for finding translation references "
                            "(e.g. '暗証番号 vs パスワード 違い' or 'two-factor authentication Japanese translation')"
                        ),
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of results (default 3, max 5)",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


class ToolExecutor:
    """Dispatches tool calls to glossary lookups, TM search, S3 Vectors retrieval,
    and locale validation."""

    def __init__(
        self,
        *,
        glossary_path: str | Path,
        retriever: Any | None = None,
        tm_path: str | Path | None = None,
        grammar_path: str | Path | None = None,
        brave_api_key: str | None = None,
    ) -> None:
        glossary_path = Path(glossary_path)
        self.glossary = self._load_glossary(glossary_path)
        self.retriever = retriever
        if tm_path is None:
            tm_path = glossary_path.parent / "translation_memory.jsonl"
        self.tm = _load_tm(Path(tm_path))
        if grammar_path is None:
            grammar_path = glossary_path.parent / "grammar_chunks.jsonl"
        self.grammar = _load_grammar(Path(grammar_path))
        self.brave_api_key = brave_api_key or os.environ.get("BRAVE_API_KEY", "")
        self.call_log: list[dict] = []

    @property
    def available_tool_names(self) -> list[str]:
        names = ["lookup_glossary", "validate_locale"]
        if self.tm:
            names.append("lookup_translation_memory")
        if self.grammar:
            names.append("lookup_grammar_pattern")
        if self.retriever is not None:
            names.append("search_knowledge_base")
        if self.brave_api_key:
            names.append("web_search")
        return names

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return OpenAI-format tool definitions for available tools only."""
        available = set(self.available_tool_names)
        return [
            t for t in TOOL_DEFINITIONS
            if t["function"]["name"] in available
        ]

    @staticmethod
    def _is_valid_glossary_entry(row: dict[str, str]) -> bool:
        """Filter out corrupted/scraped rows in the glossary CSV."""
        ja = (row.get("approved_ja") or "").strip()
        note = (row.get("usage_note") or "").strip()
        if not ja:
            return False
        if len(ja) > 30 or "Log" in ja or "freq=" in note:
            return False
        if any(c.isdigit() for c in ja[:5]):
            return False
        return True

    @staticmethod
    def _load_glossary(path: Path) -> dict[str, GlossaryEntry]:
        entries: dict[str, GlossaryEntry] = {}
        if not path.is_file():
            return entries
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not ToolExecutor._is_valid_glossary_entry(row):
                    continue
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

    def scan_source_for_glossary(self, source_en: str) -> list[dict[str, Any]]:
        """Pre-scan source text and return ALL matching glossary entries.

        Uses word-boundary regex, longest-match-first to avoid overlaps,
        and skips single-char terms to reduce noise.
        """
        source_lower = source_en.lower()
        raw_matches: list[tuple[str, GlossaryEntry]] = []
        for key, entry in self.glossary.items():
            if len(key) < 2:
                continue
            pattern = r"\b" + re.escape(key) + r"\b"
            if re.search(pattern, source_lower):
                raw_matches.append((key, entry))

        raw_matches.sort(key=lambda x: len(x[0]), reverse=True)
        seen_spans: list[tuple[int, int]] = []
        results: list[dict[str, Any]] = []

        for key, entry in raw_matches:
            pattern = r"\b" + re.escape(key) + r"\b"
            m = re.search(pattern, source_lower)
            if not m:
                continue
            start, end = m.start(), m.end()
            if any(s <= start < e or s < end <= e for s, e in seen_spans):
                continue
            seen_spans.append((start, end))
            result: dict[str, Any] = {
                "term": entry.source_term_en,
                "approved_ja": entry.approved_ja,
            }
            if entry.forbidden_variants:
                result["forbidden_variants"] = entry.forbidden_variants
            results.append(result)

        return results

    @staticmethod
    def format_glossary_context(matches: list[dict[str, Any]]) -> str:
        """Format matched glossary entries as a context block for prompts."""
        if not matches:
            return ""
        lines = []
        for m in matches:
            line = f"{m['term']} → {m['approved_ja']}"
            if m.get("forbidden_variants"):
                line += f" (NOT: {', '.join(m['forbidden_variants'])})"
            lines.append(f"- {line}")
        return (
            "## Mandatory Glossary — use these EXACT Japanese forms\n"
            + "\n".join(lines)
        )

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        if name == "lookup_glossary":
            return self._lookup_glossary(arguments["term"])
        if name == "search_knowledge_base":
            return self._search_kb(
                arguments["query"], arguments.get("top_k", 5)
            )
        if name == "lookup_translation_memory":
            return self._lookup_tm(
                arguments["sentence"],
                arguments.get("top_k", 3),
                arguments.get("threshold", 0.3),
            )
        if name == "lookup_grammar_pattern":
            return self._lookup_grammar(
                arguments["query"], arguments.get("top_k", 3)
            )
        if name == "validate_locale":
            return self._validate_locale(arguments["text"])
        if name == "web_search":
            return self._web_search(
                arguments["query"], arguments.get("count", 3)
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

    def _lookup_tm(
        self, sentence: str, top_k: int = 3, threshold: float = 0.3
    ) -> str:
        top_k = min(max(1, int(top_k)), 10)
        threshold = max(0.0, min(1.0, float(threshold)))

        scored = [
            (score, entry)
            for entry in self.tm
            if (score := _jaccard(sentence, entry.get("source_en", ""))) >= threshold
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        self.call_log.append(
            {"tool": "lookup_translation_memory", "sentence": sentence[:80], "hits": len(top)}
        )

        if not top:
            return json.dumps(
                {"found": False, "message": f"No TM matches above threshold {threshold}"},
                ensure_ascii=False,
            )

        return json.dumps(
            {
                "found": True,
                "count": len(top),
                "matches": [
                    {
                        "similarity": round(score, 3),
                        "source_en": e.get("source_en", ""),
                        "target_ja": e.get("target_ja", ""),
                        "topic": e.get("topic", ""),
                    }
                    for score, e in top
                ],
            },
            ensure_ascii=False,
        )

    def _lookup_grammar(self, query: str, top_k: int = 3) -> str:
        top_k = min(max(1, int(top_k)), 8)

        scored = [
            (score, chunk)
            for chunk in self.grammar
            if (score := _grammar_score(query, chunk)) > 0
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        # Deduplicate by section: keep only the highest-scoring chunk per section
        seen_sections: set[str] = set()
        deduped: list[tuple[float, dict]] = []
        for score, chunk in scored:
            section = chunk.get("section", "")
            if section not in seen_sections:
                seen_sections.add(section)
                deduped.append((score, chunk))
            if len(deduped) == top_k:
                break

        self.call_log.append(
            {"tool": "lookup_grammar_pattern", "query": query, "hits": len(deduped)}
        )

        if not deduped:
            return json.dumps(
                {"found": False, "message": f"No grammar entries matched '{query}'"},
                ensure_ascii=False,
            )

        return json.dumps(
            {
                "found": True,
                "count": len(deduped),
                "results": [
                    {
                        "section": chunk.get("section", ""),
                        "text": chunk.get("text", ""),
                        "chunk_id": chunk.get("chunk_id", ""),
                        "relevance": round(score, 3),
                    }
                    for score, chunk in deduped
                ],
            },
            ensure_ascii=False,
        )

    def _validate_locale(self, text: str) -> str:
        result = _validate_locale_text(text)
        parsed = json.loads(result)
        self.call_log.append(
            {
                "tool": "validate_locale",
                "valid": parsed.get("valid", True),
                "issue_count": parsed.get("issue_count", 0),
            }
        )
        return result

    def _web_search(self, query: str, count: int = 3) -> str:
        """Search the web via Brave Search API for translation verification."""
        if not self.brave_api_key:
            self.call_log.append(
                {"tool": "web_search", "query": query, "error": "no_api_key"}
            )
            return json.dumps(
                {"error": "Web search unavailable (BRAVE_API_KEY not configured)"}
            )

        import requests

        count = min(max(1, int(count)), 5)
        try:
            resp = requests.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": self.brave_api_key,
                },
                params={"q": query, "count": count},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            self.call_log.append(
                {"tool": "web_search", "query": query, "error": str(exc)}
            )
            return json.dumps({"error": f"Brave Search failed: {exc}"})

        results = []
        for item in (data.get("web", {}).get("results", []))[:count]:
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "description": item.get("description", "")[:300],
                }
            )

        self.call_log.append(
            {"tool": "web_search", "query": query, "count": len(results)}
        )
        return json.dumps(
            {"results": results, "count": len(results)}, ensure_ascii=False
        )

    def reset_log(self) -> None:
        self.call_log = []
