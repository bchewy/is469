from __future__ import annotations

# Note: these prompts are intentionally generic for the first S3 implementation.
# Once S2 retrieval is built, you can inject retrieved evidence into the user prompt.

SYSTEM_TRANSLATION_PROMPT = (
    "You are a professional English-to-Japanese translator. "
    "Translate the following English text into natural, fluent Japanese. "
    "Preserve the original meaning and tone."
)


def translation_user_prompt(
    *,
    source_en: str,
    context: str = "",
) -> str:
    if context:
        return (
            "Context (may contain relevant domain/style/glossary notes):\n"
            f"{context}\n\n"
            "Translate the following English text:\n"
            f"{source_en}"
        )
    return f"Translate the following English text:\n{source_en}"


SYSTEM_CRITIC_PROMPT = (
    "You are a strict translation QA evaluator. "
    "Score how well the Japanese translation meets adequacy (faithful meaning), "
    "fluency/grammar, and overall naturalness. "
    "Return ONLY valid JSON."
)

ERROR_LABEL_CATEGORIES = [
    "Terminology",
    "Accuracy",
    "Fluency/Grammar",
    "Style/Register",
    "Locale/Formatting",
]


def critic_user_prompt(
    *,
    source_en: str,
    candidate_ja: str,
    extra_instructions: str = "",
) -> str:
    # The model must return numeric coverage_score in [0, 1].
    # We also ask for explicit issues for easier rewrite prompts.
    return (
        f"English (source):\n{source_en}\n\n"
        f"Japanese (candidate):\n{candidate_ja}\n\n"
        f"{extra_instructions}\n\n"
        "Output JSON schema:\n"
        "{\n"
        '  "coverage_score": 0.0-1.0,\n'
        '  "has_error": true/false,\n'
        '  "issues": [string, ...],\n'
        '  "feedback": string\n'
        "}\n"
        "Return only the JSON object."
    )


SYSTEM_ERROR_CHECK_PROMPT = (
    "You are a strict English-to-Japanese translation error annotator. "
    "Judge whether the Japanese translation contains a meaningful error relative to the English source. "
    "Use only these categories when needed: "
    + ", ".join(ERROR_LABEL_CATEGORIES)
    + ". Return ONLY valid JSON."
)


def error_check_user_prompt(
    *,
    source_en: str,
    candidate_ja: str,
    context: str = "",
) -> str:
    ctx = ""
    if context:
        ctx = (
            "Retrieved context (may contain glossary or style evidence):\n"
            f"{context}\n\n"
        )

    return (
        f"{ctx}"
        f"English (source):\n{source_en}\n\n"
        f"Japanese (candidate):\n{candidate_ja}\n\n"
        "Output JSON schema:\n"
        "{\n"
        '  "has_error": true/false,\n'
        '  "severity": "none" | "minor" | "major",\n'
        '  "categories": [string, ...],\n'
        '  "rationale": string\n'
        "}\n"
        "Rules:\n"
        '- If the translation is acceptable, use "has_error": false, "severity": "none", and an empty categories list.\n'
        '- Only use categories from the allowed list.\n'
        "Return only the JSON object."
    )


SYSTEM_REWRITE_PROMPT = (
    "You are a professional translator. "
    "Revise the translation to fix the listed issues while preserving meaning."
)


def revision_user_prompt(
    *,
    source_en: str,
    previous_ja: str,
    feedback: str,
    extra_instructions: str = "",
    context: str = "",
) -> str:
    if context:
        ctx = (
            "Context (may contain relevant domain/style/glossary notes):\n"
            f"{context}\n\n"
        )
    else:
        ctx = ""

    tail = f"\n{extra_instructions}\n" if extra_instructions else ""

    return (
        f"{ctx}"
        f"English (source):\n{source_en}\n\n"
        f"Previous Japanese translation:\n{previous_ja}\n\n"
        f"QA feedback:\n{feedback}\n\n"
        f"{tail}"
        "Now provide the revised Japanese translation only (no extra commentary)."
    )

