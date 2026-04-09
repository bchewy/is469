from __future__ import annotations

import os
import time
from typing import Any

import requests


class OpenRouterClient:
    """Thin wrapper around the OpenRouter chat completions API with tool-use support."""

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        default_model: str = "anthropic/claude-sonnet-4-6",
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. "
                "Set OPENROUTER_API_KEY in .env or pass api_key=."
            )
        self.default_model = default_model

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[dict] | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                resp = requests.post(
                    self.BASE_URL, headers=headers, json=body, timeout=120
                )
                if resp.status_code == 429:
                    wait = 2 ** attempt
                    print(f"  Rate limited, retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                if "error" in data:
                    raise RuntimeError(f"OpenRouter error: {data['error']}")
                return data
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < 2:
                    time.sleep(2 ** attempt)

        raise last_exc or RuntimeError("OpenRouter request failed after retries")
