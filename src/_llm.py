"""LLM provider interface with retry logic.

Supported providers:
- OpenAI (Responses API)
- OpenRouter (Chat Completions API, streaming)
- Anthropic (Messages API)
- Gemini (Google GenAI)
"""

import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: Optional[Dict[str, Any]] = None
    reasoning: Optional[str] = None


class LLMProvider(ABC):
    @abstractmethod
    def call(self, prompt: str, **kwargs) -> LLMResponse: ...

    @abstractmethod
    def get_model_name(self) -> str: ...


class OpenAIProvider(LLMProvider):
    """OpenAI via the Responses API."""

    def __init__(self, api_key: str, model: str, timeout: float = 900.0):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, timeout=timeout, max_retries=0)
        self.model = model

    def get_model_name(self) -> str:
        return self.model

    def call(self, prompt: str, **kwargs) -> LLMResponse:
        resp = self.client.responses.create(model=self.model, input=prompt, **kwargs)
        usage = None
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.input_tokens,
                "completion_tokens": resp.usage.output_tokens,
                "total_tokens": (resp.usage.input_tokens + resp.usage.output_tokens),
            }
            if hasattr(resp.usage, "output_tokens_details") and resp.usage.output_tokens_details:
                rt = getattr(resp.usage.output_tokens_details, "reasoning_tokens", None)
                if rt:
                    usage["reasoning_tokens"] = rt

        # Extract reasoning summaries from output items
        reasoning_parts: list[str] = []
        for item in resp.output:
            if getattr(item, "type", None) == "reasoning":
                for s in getattr(item, "summary", []):
                    text = getattr(s, "text", None)
                    if text:
                        reasoning_parts.append(text)
        reasoning = "\n".join(reasoning_parts) if reasoning_parts else None

        return LLMResponse(
            content=resp.output_text or "", model=self.model, usage=usage, reasoning=reasoning,
        )


class OpenRouterProvider(LLMProvider):
    """OpenRouter via Chat Completions API (streaming)."""

    def __init__(self, api_key: str, model: str, timeout: float = 900.0, headers: Optional[Dict[str, str]] = None):
        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=timeout,
            default_headers=headers,
        )
        self.model = model

    def get_model_name(self) -> str:
        return self.model

    def call(self, prompt: str, **kwargs) -> LLMResponse:
        stream_options = kwargs.pop("stream_options", {})
        stream_options["include_usage"] = True

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            stream_options=stream_options,
            **kwargs,
        )

        parts: list[str] = []
        reasoning_parts: list[str] = []
        usage: dict | None = None

        for chunk in stream:
            if chunk.model_extra and "error" in chunk.model_extra:
                raise RuntimeError(f"OpenRouter error: {chunk.model_extra['error']}")

            if chunk.usage:
                usage = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens,
                }
                if hasattr(chunk.usage, "completion_tokens_details") and chunk.usage.completion_tokens_details:
                    rt = getattr(chunk.usage.completion_tokens_details, "reasoning_tokens", None)
                    if rt:
                        usage["reasoning_tokens"] = rt

            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta:
                if delta.content:
                    parts.append(delta.content)
                # OpenRouter sends reasoning for thinking models (DeepSeek, QwQ, etc.)
                r = getattr(delta, "reasoning", None)
                if r:
                    reasoning_parts.append(r)

        text = "".join(parts)
        if not text or not text.strip():
            raise RuntimeError("OpenRouter returned empty response")

        reasoning = "".join(reasoning_parts) if reasoning_parts else None

        return LLMResponse(content=text, model=self.model, usage=usage, reasoning=reasoning)


class AnthropicProvider(LLMProvider):
    """Anthropic Messages API."""

    def __init__(self, api_key: str, model: str):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def get_model_name(self) -> str:
        return self.model

    def call(self, prompt: str, **kwargs) -> LLMResponse:
        resp = self.client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        text_parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        thinking_parts = [b.thinking for b in resp.content if getattr(b, "type", None) == "thinking"]

        usage = None
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.input_tokens,
                "completion_tokens": resp.usage.output_tokens,
                "total_tokens": resp.usage.input_tokens + resp.usage.output_tokens,
            }

        reasoning = "\n".join(thinking_parts) if thinking_parts else None

        return LLMResponse(
            content="".join(text_parts), model=self.model, usage=usage, reasoning=reasoning,
        )


class GeminiProvider(LLMProvider):
    """Google Gemini."""

    def __init__(self, api_key: str, model: str):
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def get_model_name(self) -> str:
        return self.model

    def call(self, prompt: str, **kwargs) -> LLMResponse:
        resp = self.client.models.generate_content(model=self.model, contents=prompt, **kwargs)
        usage = None
        if resp.usage_metadata:
            m = resp.usage_metadata
            prompt_tokens = getattr(m, "prompt_token_count", 0) or 0
            completion_tokens = getattr(m, "candidates_token_count", 0) or 0
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": getattr(m, "total_token_count", 0) or (prompt_tokens + completion_tokens),
            }
            thoughts = getattr(m, "thoughts_token_count", None)
            if thoughts:
                usage["reasoning_tokens"] = thoughts

        # Extract thinking/thought parts from response
        reasoning = None
        if resp.candidates:
            thought_parts = [
                p.text for p in resp.candidates[0].content.parts
                if getattr(p, "thought", False) and p.text
            ]
            if thought_parts:
                reasoning = "\n".join(thought_parts)

        # Get text from non-thought parts only (resp.text may warn about non-text parts)
        text_parts = [
            p.text for p in resp.candidates[0].content.parts
            if not getattr(p, "thought", False) and p.text
        ] if resp.candidates else []
        content = "".join(text_parts) if text_parts else (resp.text or "")

        return LLMResponse(content=content, model=self.model, usage=usage, reasoning=reasoning)


def create_provider(
    provider: str,
    *,
    model: str,
    api_key: str,
    timeout: float = 900.0,
    headers: Optional[Dict[str, str]] = None,
) -> LLMProvider:
    """Create a provider instance by name."""
    p = provider.lower()
    if p == "openai":
        return OpenAIProvider(api_key=api_key, model=model, timeout=timeout)
    if p == "openrouter":
        return OpenRouterProvider(api_key=api_key, model=model, timeout=timeout, headers=headers)
    if p == "anthropic":
        return AnthropicProvider(api_key=api_key, model=model)
    if p == "gemini":
        return GeminiProvider(api_key=api_key, model=model)
    raise ValueError(f"Unknown provider: {provider!r}. Choose from: openai, openrouter, anthropic, gemini")


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------

def _status_code_from_exc(e: Exception) -> Optional[int]:
    sc = getattr(e, "status_code", None)
    if isinstance(sc, int):
        return sc
    resp = getattr(e, "response", None)
    return getattr(resp, "status_code", None) if resp else None


def _is_transient(e: Exception, sc: Optional[int], *, retry_timeouts: bool) -> bool:
    name = type(e).__name__.lower()
    msg = str(e).lower()
    if retry_timeouts and ("timeout" in name or "timeout" in msg):
        return True
    if "connection" in name or "connection" in msg:
        return True
    if sc in (408, 429, 500, 502, 503, 504):
        return True
    return False


def call_with_retry(
    provider: LLMProvider,
    prompt: str,
    *,
    max_retries: int = 2,
    backoff_base: float = 1.0,
    retry_timeouts: bool = False,
    **kwargs,
) -> tuple[Optional[LLMResponse], list[dict], int]:
    """Call provider with exponential-backoff retries on transient errors.

    Returns (response_or_None, error_list, attempt_count).
    """
    errors: list[dict] = []
    for attempt in range(max_retries + 1):
        try:
            return provider.call(prompt, **kwargs), errors, attempt + 1
        except Exception as e:
            sc = _status_code_from_exc(e)
            transient = _is_transient(e, sc, retry_timeouts=retry_timeouts)
            errors.append({"type": type(e).__name__, "message": str(e)[:500], "status_code": sc, "transient": transient})
            if attempt < max_retries and transient:
                time.sleep(backoff_base * (2**attempt) + random.uniform(0, 0.25 * backoff_base))
                continue
            break
    return None, errors, len(errors)
