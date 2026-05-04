"""Shared helpers for OpenAI / Azure OpenAI client construction."""

from __future__ import annotations

import os
import re
from copy import deepcopy


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} environment variable is not set")
    return value


def get_chat_client(model: str, api_version: str = "2024-12-01-preview"):
    """Return a chat-completions client for the requested model.

    GPT-5.4 routes through the Foundry/OpenAI-compatible endpoint.
    Older GPT baselines keep using Azure OpenAI for backward compatibility.
    """

    if model == "gpt-5.4":
        from openai import OpenAI

        base_url = _require_env("OPENAI_BASE_URL").rstrip("/")
        api_key = _require_env("OPENAI_API_KEY")
        return OpenAI(base_url=base_url, api_key=api_key)

    from openai import AzureOpenAI

    api_key = _require_env("AZURE_OPENAI_KEY")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    return AzureOpenAI(api_version=api_version, azure_endpoint=endpoint, api_key=api_key)


def build_chat_completion_kwargs(
    model: str,
    messages,
    max_completion_tokens: int,
    temperature: float = 0.0,
):
    """Build chat completion kwargs with conservative GPT-5.4 defaults."""

    kwargs = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
    }
    if model == "gpt-5.4":
        # Hint the backend to minimize hidden reasoning when supported.
        kwargs["reasoning_effort"] = "minimal"
    else:
        kwargs["temperature"] = temperature
    return kwargs


def is_content_filter_error(error_text: str | None) -> bool:
    if not error_text:
        return False
    lowered = error_text.lower()
    return "content_filter" in lowered or "responsibleaipolicyviolation" in lowered


_SENSITIVE_REPLACEMENTS = [
    (r"\b(child|boy|girl|kid|toddler)\b", "young person"),
    (r"\b(shirtless|topless|bare-chested)\b", "lightly clothed"),
    (r"\b(nude|nudity|naked)\b", "person with exposed body"),
    (r"\b(sex|sexual|sexually)\b", "intimate"),
    (r"\b(porn|pornography)\b", "explicit media"),
    (r"\b(rape|raped)\b", "sexual assault"),
    (r"\b(suicide|suicidal)\b", "self-harm related"),
    (r"\b(self-harm|self harm)\b", "self-injury"),
    (r"\b(kill|killing|killed)\b", "harm"),
    (r"\b(murder|murdered)\b", "violent harm"),
    (r"\b(gore|gory)\b", "graphic injury"),
    (r"\b(blood|bloody)\b", "injury"),
]


def sanitize_prompt_text(text: str) -> str:
    """Soften wording that often trips content filtering while keeping meaning."""

    cleaned = text
    for pattern, replacement in _SENSITIVE_REPLACEMENTS:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    return cleaned


def sanitize_messages(messages):
    """Return a deep-copied message list with text content softened."""

    sanitized = deepcopy(messages)
    for message in sanitized:
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = sanitize_prompt_text(content)
            continue
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                    item["text"] = sanitize_prompt_text(item["text"])
    return sanitized
