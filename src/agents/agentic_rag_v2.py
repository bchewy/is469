from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from src.agents.openrouter_client import OpenRouterClient
from src.agents.tools import ToolExecutor
from src.prompts.s3_prompts import (
    SYSTEM_CRITIC_PROMPT,
    SYSTEM_ERROR_CHECK_PROMPT,
    ERROR_LABEL_CATEGORIES,
    critic_user_prompt,
    error_check_user_prompt,
)

JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

# ---------------------------------------------------------------------------
# ReAct-style system prompts
# ---------------------------------------------------------------------------

SYSTEM_TRANSLATOR_PROMPT = (
    "You are a professional English-to-Japanese translator following a structured "
    "ReAct workflow: Reason, Act, Observe, Reflect.\n\n"
    "## Workflow\n\n"
    "### Step 1 — REASON\n"
    "Analyze the source text before doing anything:\n"
    "- Identify every specialized, technical, or UI term that might have an "
    "approved translation in the glossary.\n"
    "- Note the register (casual / formal / technical) and any formatting "
    "requirements (dates, numbers, lists).\n"
    "- Decide which tools you need. Do NOT call tools that are unavailable.\n\n"
    "### Step 2 — ACT\n"
    "Call your tools to gather context:\n"
    "- `lookup_glossary`: for each identified term, look up the approved Japanese form.\n"
    "- `lookup_translation_memory`: find previously validated translations of similar "
    "sentences — anchor your output to these when the similarity is high.\n"
    "- `lookup_grammar_pattern`: look up the correct particle, verb form, or grammatical "
    "construction when you are unsure (e.g. 'は vs が', 'polite ます form', 'conditional').\n"
    "- `search_knowledge_base`: find style rules or broader grammar patterns (only if available).\n"
    "You may call tools multiple times with different queries.\n\n"
    "### Step 3 — OBSERVE\n"
    "Review all tool results. Note which terms have mandatory approved forms "
    "and which had no glossary entry.\n\n"
    "### Step 4 — REFLECT & TRANSLATE\n"
    "Produce the Japanese translation, then call `validate_locale` on your draft "
    "to catch any date or punctuation formatting errors before finalising.\n"
    "Then silently verify:\n"
    "- Does the output contain ONLY Japanese text? No English preamble, "
    "no reasoning, no romanization, no commentary.\n"
    "- Are ALL approved glossary terms used exactly as specified?\n"
    "- Does the register match the source?\n"
    "- Did `validate_locale` return no issues? If it found issues, fix them first.\n\n"
    "CRITICAL: Your final message must contain ONLY the Japanese translation. "
    "Any English text in the final output is a critical failure.\n\n"
    "Translation rules:\n"
    "- If the glossary returns an approved_ja, you MUST use that exact form.\n"
    "- Translate English idioms into natural Japanese equivalents.\n"
    "- Match the register: casual \u2192 casual, formal \u2192 formal.\n"
    "- Never output Chinese characters in place of Japanese.\n"
    "- Use Japanese punctuation, not Western equivalents.\n"
    "- Format dates as MM\u6708DD\u65e5."
)

SYSTEM_REVISER_PROMPT = (
    "You are a professional English-to-Japanese translator revising a previous "
    "translation using a ReAct workflow.\n\n"
    "## Workflow\n\n"
    "### REASON\n"
    "Read the QA feedback carefully. Identify every specific issue to fix.\n\n"
    "### ACT\n"
    "If any issues relate to terminology, use `lookup_glossary` to verify "
    "the correct approved form. Use `lookup_translation_memory` to find "
    "validated reference translations for similar sentences. "
    "Use `lookup_grammar_pattern` for particle, verb form, or grammatical structure "
    "questions. Use `search_knowledge_base` for style questions (only if available).\n\n"
    "### OBSERVE\n"
    "Review tool results and plan your corrections.\n\n"
    "### REFLECT & REVISE\n"
    "Produce the corrected translation, then call `validate_locale` on it to "
    "confirm all formatting issues are resolved. Before outputting, verify:\n"
    "- Every issue from the feedback is addressed.\n"
    "- No new errors are introduced.\n"
    "- Output contains ONLY Japanese — no English, no commentary.\n"
    "- `validate_locale` returned no issues.\n\n"
    "CRITICAL: Output ONLY the revised Japanese translation."
)

SYSTEM_REFLECTION_PROMPT = (
    "You are a translation quality gate. You receive an English source and a "
    "Japanese translation candidate.\n\n"
    "Check the candidate against these rules IN ORDER:\n"
    "1. LANGUAGE PURITY: Does the output contain ONLY Japanese? "
    "Any English words, Chinese characters, commentary, or preamble = rewrite.\n"
    "2. COMPLETENESS: Is the full meaning of the source preserved? "
    "No omissions, no added information.\n"
    "3. PUNCTUATION: Are Japanese punctuation (\u3002\u3001\u300c\u300d\uff08\uff09) "
    "and date formats (MM\u6708DD\u65e5) used? No Western punctuation.\n"
    "4. NATURALNESS: Does it read like native Japanese?\n\n"
    "If the candidate passes all checks, output it unchanged.\n"
    "If any check fails, fix the issues and output the corrected version.\n\n"
    "CRITICAL: Output ONLY the Japanese translation — no explanations, "
    "no check results, no commentary."
)


@dataclass
class AgentTraceStep:
    step_type: str  # "initial" | "reflection" | "revision"
    candidate_ja: str
    critic_coverage_score: float
    critic_has_error: bool
    critic_feedback: str
    tool_calls: list[dict] = field(default_factory=list)


@dataclass
class ErrorCheckResult:
    has_error: bool
    severity: str
    categories: list[str]
    rationale: str
    raw_text: str


@dataclass
class AgentResult:
    translation: str
    coverage_score: float
    trace: list[AgentTraceStep]
    total_tool_calls: int
    error_check: ErrorCheckResult | None = None


def _extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    m = JSON_OBJECT_RE.search(text.strip())
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _run_with_tools(
    *,
    client: OpenRouterClient,
    executor: ToolExecutor,
    messages: list[dict],
    model: str,
    max_rounds: int = 10,
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> tuple[str, list[dict]]:
    """Run a chat loop, executing tool calls until the model produces text."""
    tool_defs = executor.get_tool_definitions()
    tool_calls_made: list[dict] = []

    for round_num in range(max_rounds):
        resp = client.chat(
            messages=messages,
            model=model,
            tools=tool_defs if tool_defs else None,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        msg = resp["choices"][0]["message"]

        assistant_msg: dict[str, Any] = {"role": "assistant"}
        if msg.get("content"):
            assistant_msg["content"] = msg["content"]
        if msg.get("tool_calls"):
            assistant_msg["tool_calls"] = msg["tool_calls"]
        messages.append(assistant_msg)

        if not msg.get("tool_calls"):
            return msg.get("content", "").strip(), tool_calls_made

        for tc in msg["tool_calls"]:
            fn_name = tc["function"]["name"]
            try:
                fn_args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                fn_args = {"query": tc["function"]["arguments"]}

            result = executor.execute(fn_name, fn_args)

            tool_calls_made.append({
                "name": fn_name,
                "arguments": fn_args,
                "result_preview": result[:300],
            })

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

        print(f"    tool round {round_num + 1}: {len(msg['tool_calls'])} call(s)")

    resp = client.chat(
        messages=messages, model=model, temperature=temperature, max_tokens=max_tokens
    )
    content = resp["choices"][0]["message"].get("content", "").strip()
    return content, tool_calls_made


def _self_reflect(
    *,
    client: OpenRouterClient,
    source_en: str,
    candidate_ja: str,
    model: str,
) -> str:
    """Self-reflection gate: catches obvious issues before the external critic."""
    messages = [
        {"role": "system", "content": SYSTEM_REFLECTION_PROMPT},
        {
            "role": "user",
            "content": (
                f"English (source):\n{source_en}\n\n"
                f"Japanese (candidate):\n{candidate_ja}"
            ),
        },
    ]
    resp = client.chat(
        messages=messages, model=model, temperature=0.1, max_tokens=2048
    )
    reflected = resp["choices"][0]["message"].get("content", "").strip()
    return reflected if reflected else candidate_ja


def _run_critic(
    *,
    client: OpenRouterClient,
    source_en: str,
    candidate_ja: str,
    model: str,
) -> tuple[float, bool, str]:
    """Score a candidate translation. Returns (coverage, has_error, feedback)."""
    messages = [
        {"role": "system", "content": SYSTEM_CRITIC_PROMPT},
        {
            "role": "user",
            "content": critic_user_prompt(
                source_en=source_en,
                candidate_ja=candidate_ja,
                extra_instructions="Be strict. Score 0-1.",
                context="",
            ),
        },
    ]
    resp = client.chat(
        messages=messages, model=model, temperature=0.1, max_tokens=512
    )
    raw = resp["choices"][0]["message"].get("content", "")
    parsed = _extract_json(raw) or {}

    score = 0.0
    try:
        score = max(0.0, min(1.0, float(parsed.get("coverage_score", 0.0))))
    except (TypeError, ValueError):
        pass

    has_error = bool(parsed.get("has_error", False))
    feedback = str(parsed.get("feedback", "")).strip()
    issues = parsed.get("issues", [])
    if isinstance(issues, list) and issues:
        feedback += "\nIssues: " + "; ".join(str(i) for i in issues)

    return score, has_error, feedback


def translate_with_agent(
    *,
    client: OpenRouterClient,
    executor: ToolExecutor,
    source_en: str,
    agent_cfg: dict[str, Any],
    models_cfg: dict[str, Any],
) -> AgentResult:
    """
    Full agentic translation pipeline with ReAct workflow:

    Phase 1 — TRANSLATE: ReAct loop (Reason→Act→Observe→Reflect) with tool use
    Phase 2 — SELF-REFLECT: Model reviews its own output, fixes obvious issues
    Phase 3 — CRITIC: External critic (potentially different model) scores output
    Phase 4 — REVISE: If score below threshold, revise with tools + reflect again
    """
    max_tool_rounds = int(agent_cfg.get("max_tool_rounds", 10))
    coverage_threshold = float(agent_cfg.get("coverage_min_threshold", 0.6))
    max_revisions = int(agent_cfg.get("max_revisions", 2))
    temperature = float(agent_cfg.get("temperature", 0.1))
    enable_reflection = bool(agent_cfg.get("enable_reflection", True))

    translator_model = models_cfg.get("translator", client.default_model)
    critic_model = models_cfg.get("critic", client.default_model)
    reflector_model = models_cfg.get("reflector", translator_model)

    trace: list[AgentTraceStep] = []
    total_tool_calls = 0

    available = executor.available_tool_names
    print(f"    available tools: {available}")

    # --- Phase 1: ReAct translation with tools ---
    executor.reset_log()
    messages = [
        {"role": "system", "content": SYSTEM_TRANSLATOR_PROMPT},
        {
            "role": "user",
            "content": f"Translate this English text into Japanese:\n\n{source_en}",
        },
    ]

    translation, tool_calls = _run_with_tools(
        client=client,
        executor=executor,
        messages=messages,
        model=translator_model,
        max_rounds=max_tool_rounds,
        temperature=temperature,
    )
    total_tool_calls += len(tool_calls)

    # --- Phase 2: Self-reflection ---
    if enable_reflection:
        print("    reflecting...")
        reflected = _self_reflect(
            client=client,
            source_en=source_en,
            candidate_ja=translation,
            model=reflector_model,
        )
        if reflected != translation:
            print("    reflection fixed issues")
            translation = reflected

    # --- Phase 3: External critic ---
    coverage, has_error, feedback = _run_critic(
        client=client,
        source_en=source_en,
        candidate_ja=translation,
        model=critic_model,
    )

    trace.append(
        AgentTraceStep(
            step_type="initial",
            candidate_ja=translation,
            critic_coverage_score=coverage,
            critic_has_error=has_error,
            critic_feedback=feedback,
            tool_calls=tool_calls,
        )
    )

    if coverage >= coverage_threshold:
        return AgentResult(
            translation=translation,
            coverage_score=coverage,
            trace=trace,
            total_tool_calls=total_tool_calls,
        )

    # --- Phase 4: Revision loop (tools + reflection available) ---
    revisions = 0
    while coverage < coverage_threshold and revisions < max_revisions:
        revisions += 1
        executor.reset_log()
        print(f"    revision {revisions}/{max_revisions}...")

        revision_messages = [
            {"role": "system", "content": SYSTEM_REVISER_PROMPT},
            {
                "role": "user",
                "content": (
                    f"English (source):\n{source_en}\n\n"
                    f"Previous translation (has issues):\n{translation}\n\n"
                    f"QA feedback:\n{feedback}\n\n"
                    "Provide the corrected Japanese translation only."
                ),
            },
        ]

        translation, tool_calls = _run_with_tools(
            client=client,
            executor=executor,
            messages=revision_messages,
            model=translator_model,
            max_rounds=max_tool_rounds,
            temperature=temperature,
        )
        total_tool_calls += len(tool_calls)

        if enable_reflection:
            print("    reflecting on revision...")
            reflected = _self_reflect(
                client=client,
                source_en=source_en,
                candidate_ja=translation,
                model=reflector_model,
            )
            if reflected != translation:
                print("    reflection fixed issues")
                translation = reflected

        coverage, has_error, feedback = _run_critic(
            client=client,
            source_en=source_en,
            candidate_ja=translation,
            model=critic_model,
        )

        trace.append(
            AgentTraceStep(
                step_type="revision",
                candidate_ja=translation,
                critic_coverage_score=coverage,
                critic_has_error=has_error,
                critic_feedback=feedback,
                tool_calls=tool_calls,
            )
        )

        if coverage >= coverage_threshold:
            break

    return AgentResult(
        translation=translation,
        coverage_score=coverage,
        trace=trace,
        total_tool_calls=total_tool_calls,
    )


def detect_error_with_api(
    *,
    client: OpenRouterClient,
    source_en: str,
    candidate_ja: str,
    model: str,
) -> ErrorCheckResult:
    """Classify translation errors using the API."""
    messages = [
        {"role": "system", "content": SYSTEM_ERROR_CHECK_PROMPT},
        {
            "role": "user",
            "content": error_check_user_prompt(
                source_en=source_en,
                candidate_ja=candidate_ja,
                context="",
            ),
        },
    ]
    resp = client.chat(
        messages=messages, model=model, temperature=0.1, max_tokens=256
    )
    raw = resp["choices"][0]["message"].get("content", "")
    parsed = _extract_json(raw) or {}

    has_error = bool(parsed.get("has_error", False))
    severity = str(parsed.get("severity", "none") or "none").strip().lower()
    if severity not in {"none", "minor", "major"}:
        severity = "none" if not has_error else "minor"

    cats_raw = parsed.get("categories", [])
    categories = [
        str(c).strip()
        for c in (cats_raw if isinstance(cats_raw, list) else [])
        if str(c).strip() in ERROR_LABEL_CATEGORIES
    ]

    rationale = str(parsed.get("rationale", "")).strip()

    if not has_error:
        severity = "none"
        categories = []

    return ErrorCheckResult(
        has_error=has_error,
        severity=severity,
        categories=categories,
        rationale=rationale,
        raw_text=raw,
    )
