from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.prompts.s3_prompts import (
    ERROR_LABEL_CATEGORIES,
    SYSTEM_CRITIC_PROMPT,
    SYSTEM_ERROR_CHECK_PROMPT,
    SYSTEM_REWRITE_PROMPT,
    SYSTEM_TRANSLATION_PROMPT,
    critic_user_prompt,
    error_check_user_prompt,
    revision_user_prompt,
    translation_user_prompt,
)


JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

# ---------------------------------------------------------------------------
# Error category constants — used for routing decisions inside the loop
# ---------------------------------------------------------------------------
CAT_TERMINOLOGY = "Terminology"
CAT_ACCURACY = "Accuracy"
CAT_FLUENCY = "Fluency/Grammar"
CAT_STYLE = "Style/Register"
CAT_LOCALE = "Locale/Formatting"


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """
    Best-effort JSON extraction:
    - find the first JSON object in the text
    - parse it
    """
    if not text:
        return None

    match = JSON_OBJECT_RE.search(text.strip())
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CriticResult:
    coverage_score: float
    has_error: bool
    issues: list[str]
    feedback: str
    raw_text: str

    @staticmethod
    def from_model_output(raw_text: str) -> "CriticResult":
        parsed = _extract_json_object(raw_text) or {}

        def _as_float(v: Any, default: float = 0.0) -> float:
            try:
                return float(v)
            except Exception:
                return default

        coverage_score = _as_float(parsed.get("coverage_score"), default=0.0)
        has_error = bool(parsed.get("has_error", False))

        issues_raw = parsed.get("issues", [])
        issues = [str(x) for x in issues_raw] if isinstance(issues_raw, list) else []

        feedback = str(parsed.get("feedback", "")).strip()
        coverage_score = max(0.0, min(1.0, coverage_score))

        return CriticResult(
            coverage_score=coverage_score,
            has_error=has_error,
            issues=issues,
            feedback=feedback,
            raw_text=raw_text,
        )


@dataclass
class ErrorCheckResult:
    has_error: bool
    severity: str
    categories: list[str]
    rationale: str
    raw_text: str

    @staticmethod
    def from_model_output(raw_text: str) -> "ErrorCheckResult":
        parsed = _extract_json_object(raw_text) or {}

        has_error = bool(parsed.get("has_error", False))
        severity = str(parsed.get("severity", "none") or "none").strip().lower()
        if severity not in {"none", "minor", "major"}:
            severity = "none" if not has_error else "minor"

        categories_raw = parsed.get("categories", [])
        if isinstance(categories_raw, list):
            categories = [
                str(x).strip()
                for x in categories_raw
                if str(x).strip() in ERROR_LABEL_CATEGORIES
            ]
        else:
            categories = []

        rationale = str(parsed.get("rationale", "")).strip()

        if not has_error:
            severity = "none"
            categories = []

        return ErrorCheckResult(
            has_error=has_error,
            severity=severity,
            categories=categories,
            rationale=rationale,
            raw_text=raw_text,
        )


@dataclass
class AgentTraceStep:
    step_type: str          # "initial" | "rewrite" | "revision"
    candidate_ja: str
    critic_coverage_score: float
    critic_has_error: bool
    critic_feedback: str
    # NEW: track what the error classifier found and what strategy was chosen
    error_categories: list[str] = field(default_factory=list)
    strategy_used: str = ""         # "terminology_reretrieval" | "fluency_rewrite" | "locale_correction" | "general_rewrite"
    context_refreshed: bool = False # whether retrieval was re-run this step


# ---------------------------------------------------------------------------
# Core generation helper (unchanged)
# ---------------------------------------------------------------------------

def _generate_chat(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    num_beams: int,
) -> str:
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    generation_kwargs: dict[str, Any] = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "num_beams": max(1, num_beams),
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.no_grad():
        output_ids = model.generate(**generation_kwargs)

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_critic(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    source_en: str,
    candidate_ja: str,
    context: str,
    extra_instructions: str,
    gen_cfg: dict[str, Any],
) -> CriticResult:
    """Run the critic and return a parsed CriticResult."""
    temperature = float(gen_cfg.get("temperature", 0.1))
    top_p = float(gen_cfg.get("top_p", 0.95))
    do_sample = bool(gen_cfg.get("do_sample", False))
    num_beams = int(gen_cfg.get("num_beams", 1))

    messages = [
        {"role": "system", "content": SYSTEM_CRITIC_PROMPT},
        {
            "role": "user",
            "content": critic_user_prompt(
                source_en=source_en,
                candidate_ja=candidate_ja,
                extra_instructions=extra_instructions,
                context=context,
            ),
        },
    ]
    raw = _generate_chat(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_new_tokens=256,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        num_beams=num_beams,
    )
    return CriticResult.from_model_output(raw)


def _run_error_check(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    source_en: str,
    candidate_ja: str,
    context: str,
    gen_cfg: dict[str, Any],
) -> ErrorCheckResult:
    """Run error classification and return a parsed ErrorCheckResult."""
    temperature = float(gen_cfg.get("temperature", 0.1))
    top_p = float(gen_cfg.get("top_p", 0.95))
    do_sample = bool(gen_cfg.get("do_sample", False))
    num_beams = int(gen_cfg.get("num_beams", 1))

    messages = [
        {"role": "system", "content": SYSTEM_ERROR_CHECK_PROMPT},
        {
            "role": "user",
            "content": error_check_user_prompt(
                source_en=source_en,
                candidate_ja=candidate_ja,
                context=context,
            ),
        },
    ]
    raw = _generate_chat(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_new_tokens=256,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        num_beams=num_beams,
    )
    return ErrorCheckResult.from_model_output(raw)


def _choose_strategy(error_categories: list[str]) -> str:
    """
    Route to a repair strategy based on the error classifier's output.
    Priority order reflects which errors benefit most from different strategies.
    """
    if CAT_TERMINOLOGY in error_categories:
        return "terminology_reretrieval"
    if CAT_FLUENCY in error_categories or CAT_ACCURACY in error_categories:
        return "fluency_rewrite"
    if CAT_LOCALE in error_categories:
        return "locale_correction"
    return "general_rewrite"


def _build_strategy_instructions(
    strategy: str,
    critic_feedback: str,
    critic_issues: list[str],
    error_categories: list[str],
) -> tuple[str, str]:
    """
    Return (feedback, extra_instructions) tailored to the chosen strategy.
    """
    base_issues = (
        "Issues to address:\n- " + "\n- ".join(critic_issues)
        if critic_issues
        else ""
    )

    if strategy == "terminology_reretrieval":
        feedback = (
            critic_feedback
            or "Terminology is incorrect. Use the glossary entries in the context to fix term usage."
        )
        extra = (
            base_issues + "\nFocus specifically on correct term usage from the retrieved glossary context."
        )

    elif strategy == "fluency_rewrite":
        feedback = (
            critic_feedback
            or "The translation has fluency or accuracy issues. Rewrite for natural Japanese and faithful meaning."
        )
        extra = base_issues + "\nPrioritise grammatical naturalness and meaning accuracy."

    elif strategy == "locale_correction":
        feedback = (
            critic_feedback
            or "Locale/formatting is wrong. Fix date formats, number formats, and Japanese punctuation."
        )
        extra = (
            base_issues
            + "\nEnsure dates use Japanese conventions (年月日), "
            "numbers use full-width where appropriate, and punctuation is correct (。、「」)."
        )

    else:  # general_rewrite
        feedback = critic_feedback or "Improve translation quality and fluency."
        extra = base_issues

    return feedback, extra


# ---------------------------------------------------------------------------
# Main agentic loop — now truly agentic
# ---------------------------------------------------------------------------

def translate_with_agentic_loop(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    source_en: str,
    context: str,
    agent_cfg: dict[str, Any],
    gen_cfg: dict[str, Any],
    # NEW: optional retriever callable — pass in to enable dynamic re-retrieval
    # Signature: retriever(query: str) -> str
    # If None, context stays static (backwards-compatible with S2 behaviour)
    retriever: Callable[[str], str] | None = None,
) -> tuple[str, float, list[AgentTraceStep]]:
    """
    Agentic S3 translation loop with:
    - Dynamic re-retrieval when terminology errors are detected (Gap 1)
    - Error-type routing: different repair strategies per MQM category (Gap 2)
    - Error classifier runs INSIDE the loop to drive decisions (Gap 3)
    """
    rewrite_retry_limit = int(agent_cfg.get("rewrite_retry_limit", 1))
    revision_limit = int(agent_cfg.get("revision_limit", 1))
    coverage_min_threshold = float(agent_cfg.get("coverage_min_threshold", 0.6))

    max_new_tokens = int(gen_cfg.get("max_new_tokens", 512))
    temperature = float(gen_cfg.get("temperature", 0.1))
    top_p = float(gen_cfg.get("top_p", 0.95))
    do_sample = bool(gen_cfg.get("do_sample", False))
    num_beams = int(gen_cfg.get("num_beams", 1))

    trace: list[AgentTraceStep] = []
    rewrites = 0
    revisions = 0

    # ------------------------------------------------------------------
    # Step 1: Initial translation
    # ------------------------------------------------------------------
    messages = [
        {"role": "system", "content": SYSTEM_TRANSLATION_PROMPT},
        {
            "role": "user",
            "content": translation_user_prompt(source_en=source_en, context=context),
        },
    ]
    candidate = _generate_chat(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        num_beams=num_beams,
    )

    # ------------------------------------------------------------------
    # Step 2: Initial critic
    # ------------------------------------------------------------------
    critic = _run_critic(
        model=model,
        tokenizer=tokenizer,
        source_en=source_en,
        candidate_ja=candidate,
        context=context,
        extra_instructions="Assess adequacy and fluency. Be strict.",
        gen_cfg=gen_cfg,
    )
    trace.append(
        AgentTraceStep(
            step_type="initial",
            candidate_ja=candidate,
            critic_coverage_score=critic.coverage_score,
            critic_has_error=critic.has_error,
            critic_feedback=critic.feedback,
        )
    )

    if critic.coverage_score >= coverage_min_threshold:
        return candidate, critic.coverage_score, trace

    # ------------------------------------------------------------------
    # Step 3: Agentic repair loop
    # ------------------------------------------------------------------
    while (
        critic.coverage_score < coverage_min_threshold
        and (rewrites < rewrite_retry_limit or revisions < revision_limit)
    ):
        step_type = "rewrite" if rewrites < rewrite_retry_limit else "revision"
        if step_type == "rewrite":
            rewrites += 1
        else:
            revisions += 1

        # --- GAP 2 & 3 FIX: run error classifier INSIDE the loop ---
        error_check = _run_error_check(
            model=model,
            tokenizer=tokenizer,
            source_en=source_en,
            candidate_ja=candidate,
            context=context,
            gen_cfg=gen_cfg,
        )
        error_categories = error_check.categories

        # --- Route to a repair strategy based on error type ---
        strategy = _choose_strategy(error_categories)

        # --- GAP 1 FIX: dynamic re-retrieval for terminology errors ---
        context_refreshed = False
        if strategy == "terminology_reretrieval" and retriever is not None:
            # Re-query the KB using the critic's feedback as the search query
            # This targets the specific problematic terminology
            targeted_query = critic.feedback if critic.feedback else source_en
            new_context = retriever(targeted_query)
            if new_context:
                context = new_context
                context_refreshed = True

        # --- Build strategy-specific instructions ---
        feedback, extra_instructions = _build_strategy_instructions(
            strategy=strategy,
            critic_feedback=critic.feedback,
            critic_issues=critic.issues,
            error_categories=error_categories,
        )

        # --- Generate revised translation ---
        revision_messages = [
            {"role": "system", "content": SYSTEM_REWRITE_PROMPT},
            {
                "role": "user",
                "content": revision_user_prompt(
                    source_en=source_en,
                    previous_ja=candidate,
                    feedback=feedback,
                    extra_instructions=extra_instructions,
                    context=context,
                ),
            },
        ]
        candidate = _generate_chat(
            model=model,
            tokenizer=tokenizer,
            messages=revision_messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            num_beams=num_beams,
        )

        # --- Re-score the revised candidate ---
        critic = _run_critic(
            model=model,
            tokenizer=tokenizer,
            source_en=source_en,
            candidate_ja=candidate,
            context=context,
            extra_instructions="Re-evaluate the revised translation. Strictly score 0-1.",
            gen_cfg=gen_cfg,
        )

        trace.append(
            AgentTraceStep(
                step_type=step_type,
                candidate_ja=candidate,
                critic_coverage_score=critic.coverage_score,
                critic_has_error=critic.has_error,
                critic_feedback=critic.feedback,
                error_categories=error_categories,
                strategy_used=strategy,
                context_refreshed=context_refreshed,
            )
        )

        if critic.coverage_score >= coverage_min_threshold:
            break

    return candidate, critic.coverage_score, trace


# ---------------------------------------------------------------------------
# Standalone error detection (called from run_s3.py after the loop for metrics)
# ---------------------------------------------------------------------------

def detect_translation_error(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    source_en: str,
    candidate_ja: str,
    context: str,
    gen_cfg: dict[str, Any],
) -> ErrorCheckResult:
    return _run_error_check(
        model=model,
        tokenizer=tokenizer,
        source_en=source_en,
        candidate_ja=candidate_ja,
        context=context,
        gen_cfg=gen_cfg,
    )