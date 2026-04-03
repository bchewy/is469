from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

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


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """
    Best-effort JSON extraction:
    - find the first JSON object in the text
    - parse it
    """
    if not text:
        return None

    # Common failure mode: model wraps JSON in extra text.
    match = JSON_OBJECT_RE.search(text.strip())
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except Exception:
        return None


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
        if isinstance(issues_raw, list):
            issues = [str(x) for x in issues_raw]
        else:
            issues = []

        feedback = str(parsed.get("feedback", "")).strip()

        # Clamp to [0, 1] for safety.
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
            categories = [str(x).strip() for x in categories_raw if str(x).strip() in ERROR_LABEL_CATEGORIES]
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
    step_type: str  # "initial" | "rewrite" | "revision"
    candidate_ja: str
    critic_coverage_score: float
    critic_has_error: bool
    critic_feedback: str


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

    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def translate_with_agentic_loop(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    source_en: str,
    context: str,
    agent_cfg: dict[str, Any],
    gen_cfg: dict[str, Any],
) -> tuple[str, float, list[AgentTraceStep]]:
    """
    Initial S3 version:
    - retrieval context is passed in, but can be empty
    - agent uses a critic JSON score + feedback
    - then rewrites/revises until coverage >= threshold or limits reached
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

    # Step 1: initial candidate
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

    # Step 1 critic
    critic_messages = [
        {"role": "system", "content": SYSTEM_CRITIC_PROMPT},
        {
            "role": "user",
            "content": critic_user_prompt(
                source_en=source_en,
                candidate_ja=candidate,
                extra_instructions="Assess adequacy and fluency. Be strict.",
                context=context,
            ),
        },
    ]
    critic_raw = _generate_chat(
        model=model,
        tokenizer=tokenizer,
        messages=critic_messages,
        max_new_tokens=256,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        num_beams=num_beams,
    )
    critic = CriticResult.from_model_output(critic_raw)
    trace.append(
        AgentTraceStep(
            step_type="initial",
            candidate_ja=candidate,
            critic_coverage_score=critic.coverage_score,
            critic_has_error=critic.has_error,
            critic_feedback=critic.feedback,
        )
    )

    # If it is already good enough, stop.
    if critic.coverage_score >= coverage_min_threshold:
        return candidate, critic.coverage_score, trace

    # Step 2+: agent loop.
    while (
        critic.coverage_score < coverage_min_threshold
        and (rewrites < rewrite_retry_limit or revisions < revision_limit)
    ):
        if rewrites < rewrite_retry_limit:
            step_type = "rewrite"
            rewrites += 1
        else:
            step_type = "revision"
            revisions += 1

        feedback = critic.feedback or "Improve translation quality and fluency."
        extra_instructions = ""
        if critic.issues:
            extra_instructions = "Issues to address:\n- " + "\n- ".join(critic.issues)

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

        # Rebuild critic prompt for the new candidate.
        critic_messages = [
            {"role": "system", "content": SYSTEM_CRITIC_PROMPT},
            {
                "role": "user",
                "content": critic_user_prompt(
                    source_en=source_en,
                    candidate_ja=candidate,
                    extra_instructions="Re-evaluate the revised translation. Strictly score 0-1.",
                    context=context,
                ),
            },
        ]
        critic_raw = _generate_chat(
            model=model,
            tokenizer=tokenizer,
            messages=critic_messages,
            max_new_tokens=256,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            num_beams=num_beams,
        )
        critic = CriticResult.from_model_output(critic_raw)

        trace.append(
            AgentTraceStep(
                step_type=step_type,
                candidate_ja=candidate,
                critic_coverage_score=critic.coverage_score,
                critic_has_error=critic.has_error,
                critic_feedback=critic.feedback,
            )
        )

        if critic.coverage_score >= coverage_min_threshold:
            break

    return candidate, critic.coverage_score, trace


def detect_translation_error(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    source_en: str,
    candidate_ja: str,
    context: str,
    gen_cfg: dict[str, Any],
) -> ErrorCheckResult:
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

