"""Shared Gemini API helpers for verifier fallbacks."""

from __future__ import annotations

import json
import os
import re
from urllib import error as urlerror
from urllib import request as urlrequest

from ._regexes import FENCE_END_PATTERN, FENCE_START_PATTERN


def gemini_api_key() -> str | None:
    """Return configured Gemini API key, if available."""
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def strip_json_fence(text: str) -> str:
    """Strip optional markdown code fences around a JSON string."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(FENCE_START_PATTERN, "", stripped)
        stripped = re.sub(FENCE_END_PATTERN, "", stripped)
    return stripped.strip()


def call_gemini_json(*, prompt: str, model: str, timeout_s: float) -> dict[str, object]:
    """Call Gemini `generateContent` and parse a JSON-object response body."""
    api_key = gemini_api_key()
    if not api_key:
        raise RuntimeError("Gemini API key not found (set GEMINI_API_KEY or GOOGLE_API_KEY)")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }
    request = urlrequest.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlrequest.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
    except urlerror.URLError as exc:
        raise RuntimeError("Gemini request failed") from exc

    response_payload = json.loads(raw)
    candidates = response_payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Gemini response missing candidates")

    content = candidates[0].get("content")
    if not isinstance(content, dict):
        raise ValueError("Gemini response missing content")

    parts = content.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError("Gemini response missing parts")

    text_chunks: list[str] = []
    for part in parts:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            text_chunks.append(part["text"])

    parsed = json.loads(strip_json_fence("".join(text_chunks)))
    if not isinstance(parsed, dict):
        raise ValueError("Gemini fallback response is not a JSON object")
    return parsed
