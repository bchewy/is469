from __future__ import annotations

SYSTEM_TRANSLATION_PROMPT = (
    "You are a professional English-to-Japanese translator. "
    "Translate the following English text into natural, fluent Japanese. "
    "Preserve the original meaning, tone, and register exactly.\n\n"
    "Rules:\n"
    "- Output ONLY the Japanese translation. No explanations, no romanization, no commentary.\n"
    "- If the context block contains glossary entries in the format "
    "'EN_TERM → APPROVED_JA', you MUST use the approved Japanese form for that term.\n"
    "- Translate English idioms into natural Japanese equivalents, not word-for-word.\n"
    "- Match the register of the source: casual sources get casual Japanese, "
    "formal sources get formal Japanese.\n"
    "- Never output Chinese characters (simplified or traditional) in place of Japanese."
)


def translation_user_prompt(
    *,
    source_en: str,
    context: str = "",
) -> str:
    if context:
        return (
            "Context — use any glossary terms (EN_TERM → APPROVED_JA) listed below:\n"
            f"{context}\n\n"
            "---\n"
            "Translate this English text into Japanese:\n"
            f"{source_en}"
        )
    return f"Translate this English text into Japanese:\n{source_en}"


SYSTEM_CRITIC_PROMPT = (
    "You are a strict translation QA evaluator for English-to-Japanese translation. "
    "You MUST check ALL of the following before scoring:\n\n"
    "1. LANGUAGE PURITY: Does the output contain ONLY Japanese text? "
    "Flag ANY simplified Chinese characters (e.g. 不该, 几乎, 吗, 过, 们), "
    "English words left untranslated, or any other non-Japanese content as a MAJOR error. "
    "Score 0.0 immediately if found.\n\n"
    "2. GLOSSARY COMPLIANCE: If the context contains glossary entries in the format "
    "'EN_TERM → APPROVED_JA', does the translation use the exact APPROVED_JA form? "
    "Flag any deviation as an error.\n\n"
    "3. ACCURACY: Does the translation preserve the full meaning of the source? "
    "Check for omitted information, added meaning, reversed roles, or factual errors.\n\n"
    "4. NATURALNESS: Does the translation sound like natural native Japanese? "
    "Check for unnatural particle usage, awkward phrasing, wrong verb forms, "
    "and literal word-for-word translation patterns.\n\n"
    "5. IDIOMS: Are English idioms and set phrases translated to natural Japanese equivalents, "
    "not translated literally?\n\n"
    "6. REGISTER: Does the formality level match the source? "
    "Casual English should not become formal keigo, and vice versa.\n\n"
    "Be STRICT. A score above 0.85 should be uncommon. "
    "Return ONLY a valid JSON object — no explanation, no extra text."
)


ERROR_LABEL_CATEGORIES = [
    "Terminology",
    "Accuracy",
    "Fluency/Grammar",
    "Style/Register",
    "Locale/Formatting",
]

_CATEGORY_DEFINITIONS = (
    "Category definitions:\n"
    "- Terminology: Wrong or inconsistent word choice for a specific term, "
    "untranslated English words left in the output, or use of a forbidden variant "
    "instead of the approved Japanese form. "
    "Example: translating 'donkey' as 馬 (horse) instead of ロバ.\n"
    "- Accuracy: The meaning of the source is changed, omitted, or distorted. "
    "Includes wrong numbers, reversed roles, missed negation, or misread idioms. "
    "Example: 'I was not happy' translated as '私は幸せでした' (I was happy).\n"
    "- Fluency/Grammar: The Japanese is grammatically incorrect or unnatural. "
    "Includes wrong particles (は vs が), incorrect verb conjugation, unnatural word order, "
    "or phrasing that a native speaker would never use. "
    "Example: '母を世話する' instead of '母の世話をする'.\n"
    "- Style/Register: The formality level does not match the source. "
    "Includes using plain form (だ/である) when polite form (です/ます) is expected, "
    "or vice versa. Also covers inappropriate honorific usage or overly stiff/casual tone. "
    "Example: translating a casual request as overly formal keigo.\n"
    "- Locale/Formatting: Wrong format for numbers, dates, currencies, punctuation, "
    "or proper nouns. Includes using Western punctuation instead of Japanese 。、"
    ", or leaving proper nouns in English when a standard Japanese form exists.\n"
)


def critic_user_prompt(
    *,
    source_en: str,
    candidate_ja: str,
    extra_instructions: str = "",
    context: str = "",
) -> str:
    ctx_block = ""
    if context:
        ctx_block = (
            "Retrieved glossary/TM context (verify approved terms were used):\n"
            f"{context}\n\n"
        )

    extra = f"{extra_instructions}\n\n" if extra_instructions else ""

    return (
        f"{ctx_block}"
        f"English (source):\n{source_en}\n\n"
        f"Japanese (candidate):\n{candidate_ja}\n\n"
        f"{extra}"
        "Checklist before scoring:\n"
        "- Does the candidate contain ANY non-Japanese text (simplified Chinese, "
        "English, etc.)? If yes → score 0.0, has_error: true.\n"
        "- If glossary terms were retrieved, did the candidate use the approved Japanese form?\n"
        "- Is the Japanese natural and idiomatic?\n"
        "- Is the meaning fully preserved?\n"
        "- Does the register match the source?\n\n"
        "Output JSON schema:\n"
        "{\n"
        '  "coverage_score": 0.0-1.0,\n'
        '  "has_error": true/false,\n'
        '  "issues": [string, ...],\n'
        '  "feedback": string\n'
        "}\n"
        "Return ONLY the JSON object."
    )


SYSTEM_ERROR_CHECK_PROMPT = (
    "You are a strict English-to-Japanese translation error annotator. "
    "Your task is to judge whether the Japanese translation contains a meaningful error "
    "relative to the English source, and classify it using the categories below.\n\n"
    + _CATEGORY_DEFINITIONS
    + "\nImportant rules:\n"
    "- has_error must be true if there is ANY error, even minor.\n"
    "- Only assign categories that directly apply. Do not assign all categories by default.\n"
    "- Fluency/Grammar applies when the Japanese itself is ungrammatical or unnatural, "
    "regardless of meaning.\n"
    "- Style/Register applies when formality is wrong even if the meaning is correct.\n"
    "- Locale/Formatting applies when punctuation, number format, or proper noun form is wrong.\n"
    "Return ONLY a valid JSON object — no explanation, no extra text, no Japanese output."
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
        "Evaluate the candidate translation. Check in this order:\n"
        "1. Does the candidate contain non-Japanese text? → Terminology or Accuracy error.\n"
        "2. Is the meaning preserved? → Accuracy if not.\n"
        "3. Is the Japanese grammatically natural? → Fluency/Grammar if not.\n"
        "4. Does the formality level match the source? → Style/Register if not.\n"
        "5. Are numbers, dates, and punctuation in correct Japanese format? "
        "→ Locale/Formatting if not.\n\n"
        "Output JSON schema:\n"
        "{\n"
        '  "has_error": true/false,\n'
        '  "severity": "none" | "minor" | "major",\n'
        '  "categories": [only from: Terminology, Accuracy, Fluency/Grammar, '
        'Style/Register, Locale/Formatting],\n'
        '  "rationale": "one sentence explaining the main issue, or why there is none"\n'
        "}\n"
        "Rules:\n"
        "- severity must be 'none' when has_error is false.\n"
        "- severity 'major' = meaning is significantly wrong or output is not Japanese.\n"
        "- severity 'minor' = meaning is mostly correct but phrasing or style is off.\n"
        "- categories must be an empty list when has_error is false.\n"
        "Return ONLY the JSON object."
    )


SYSTEM_REWRITE_PROMPT = (
    "You are a professional English-to-Japanese translator. "
    "You will be given a previous translation and specific QA feedback. "
    "Revise the translation to fix every listed issue while preserving the original meaning exactly.\n\n"
    "Rules:\n"
    "- Output ONLY the revised Japanese translation. No commentary.\n"
    "- If the context contains glossary entries (EN_TERM → APPROVED_JA), "
    "use the approved Japanese form.\n"
    "- Fix the specific issues mentioned in the feedback. Do not introduce new errors.\n"
    "- Never output Chinese characters or untranslated English words."
)


def revision_user_prompt(
    *,
    source_en: str,
    previous_ja: str,
    feedback: str,
    extra_instructions: str = "",
    context: str = "",
) -> str:
    ctx = ""
    if context:
        ctx = (
            "Context — use any glossary terms (EN_TERM → APPROVED_JA) listed below:\n"
            f"{context}\n\n"
            "---\n"
        )

    tail = f"\nAdditional issues to fix:\n{extra_instructions}\n" if extra_instructions else ""

    return (
        f"{ctx}"
        f"English (source):\n{source_en}\n\n"
        f"Previous Japanese translation (contains errors):\n{previous_ja}\n\n"
        f"QA feedback:\n{feedback}\n"
        f"{tail}\n"
        "Provide the corrected Japanese translation only:"
    )