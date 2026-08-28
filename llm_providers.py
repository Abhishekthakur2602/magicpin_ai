"""
Pluggable LLM providers for the Vera composer.

Set via environment variables:
    LLM_PROVIDER = "anthropic" | "openai" | "gemini" | "deepseek" | "groq" | "cerebras" | "none"
    LLM_API_KEY  = "<your key>"
    LLM_MODEL    = "<optional override>"
    GROQ_REASONING_EFFORT = "low" | "medium" | "high" | "none"  (Groq reasoning models only, default "low")

If LLM_PROVIDER is unset, missing a key, or the call fails/times out, the
composer falls back to the deterministic rule-based template engine in
composer.py — so the bot degrades gracefully instead of crashing a tick.

All providers are called with temperature=0 to keep compose() deterministic
given the same inputs, as required by the challenge contract.

Two things worth knowing if you're extending this:

1. Every request sets an explicit User-Agent. Several providers (Groq
   included) sit behind Cloudflare/WAF edge protection that silently 403s
   Python's default "Python-urllib/x.y" user agent even when the API key
   and payload are completely valid — curl works because it sends a normal
   UA. Omitting this header is a common, confusing source of "works in
   curl, 403 in code" bugs.

2. complete() accepts an optional `timeout` in seconds. The judge harness
   gives /v1/tick a hard 30s budget for up to 20 actions, so bot.py computes
   a shrinking per-call timeout as it works through a tick rather than
   always requesting the full TIMEOUT_LLM window — a single slow call can't
   then blow through the whole tick's budget regardless of when it started.
"""

from __future__ import annotations

import json
import os
import re
import time as _time
from abc import ABC, abstractmethod
from urllib import request as urlrequest, error as urlerror

TIMEOUT_LLM = 20  # default/ceiling — callers may pass a shorter timeout

_DEFAULT_HEADERS = {
    "User-Agent": "vera-bot/1.0 (+https://magicpin.in)",
    "Content-Type": "application/json",
}


def _post_json(url: str, body: dict, extra_headers: dict, timeout: float, retries: int = 1) -> dict:
    """POST JSON, raise a RuntimeError with the actual response body on
    failure (instead of a bare HTTPError) so callers can see *why* a call
    failed rather than just the status code.

    Retries once (by default) on HTTP 429, parsing the provider's suggested
    wait time out of the error body when present (Groq/OpenAI both include
    a "try again in Xs" hint) — but only if there's actually enough of the
    caller's timeout budget left to wait and retry; otherwise it fails fast
    so the caller (composer.py) can fall back to the rule-based path well
    within the judge's 30s tick deadline."""
    headers = {**_DEFAULT_HEADERS, **extra_headers}
    req = urlrequest.Request(
        url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
    )
    start = _time.monotonic()
    try:
        resp = urlrequest.urlopen(req, timeout=timeout)
        return json.loads(resp.read().decode("utf-8"))
    except urlerror.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8")[:500]
        except Exception:
            pass
        if e.code == 429 and retries > 0:
            wait_s = 2.0
            m = re.search(r"try again in ([\d.]+)s", detail, re.IGNORECASE)
            if m:
                wait_s = float(m.group(1)) + 0.5
            elapsed = _time.monotonic() - start
            remaining = timeout - elapsed
            if wait_s < remaining - 1:  # only retry if we can still finish in time
                _time.sleep(wait_s)
                return _post_json(url, body, extra_headers, timeout=remaining - wait_s, retries=retries - 1)
        raise RuntimeError(f"HTTP {e.code} from {url}: {detail}") from e


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, prompt: str, system: str | None = None, timeout: float | None = None) -> str: ...

    @abstractmethod
    def name(self) -> str: ...


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = ""):
        self.api_key = api_key
        self.model = model or "claude-sonnet-4-6"

    def name(self) -> str:
        return f"Anthropic ({self.model})"

    def complete(self, prompt: str, system: str | None = None, timeout: float | None = None) -> str:
        body = {
            "model": self.model,
            "max_tokens": 400,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system
        data = _post_json(
            "https://api.anthropic.com/v1/messages",
            body,
            {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
            timeout=timeout or TIMEOUT_LLM,
        )
        return data["content"][0]["text"]


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = ""):
        self.api_key = api_key
        self.model = model or "gpt-4o-mini"

    def name(self) -> str:
        return f"OpenAI ({self.model})"

    def complete(self, prompt: str, system: str | None = None, timeout: float | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        data = _post_json(
            "https://api.openai.com/v1/chat/completions",
            {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": 400},
            {"Authorization": f"Bearer {self.api_key}"},
            timeout=timeout or TIMEOUT_LLM,
        )
        return data["choices"][0]["message"]["content"]


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = ""):
        self.api_key = api_key
        self.model = model or "gemini-1.5-flash"

    def name(self) -> str:
        return f"Gemini ({self.model})"

    def complete(self, prompt: str, system: str | None = None, timeout: float | None = None) -> str:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self.model}:generateContent?key={self.api_key}")
        data = _post_json(
            url,
            {"contents": [{"parts": [{"text": full_prompt}]}],
             "generationConfig": {"temperature": 0, "maxOutputTokens": 400}},
            {},
            timeout=timeout or TIMEOUT_LLM,
        )
        return data["candidates"][0]["content"]["parts"][0]["text"]


class DeepSeekProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = ""):
        self.api_key = api_key
        self.model = model or "deepseek-chat"

    def name(self) -> str:
        return f"DeepSeek ({self.model})"

    def complete(self, prompt: str, system: str | None = None, timeout: float | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        data = _post_json(
            "https://api.deepseek.com/v1/chat/completions",
            {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": 400},
            {"Authorization": f"Bearer {self.api_key}"},
            timeout=timeout or TIMEOUT_LLM,
        )
        return data["choices"][0]["message"]["content"]


class GroqProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = ""):
        self.api_key = api_key
        self.model = model or "openai/gpt-oss-120b"

    def name(self) -> str:
        return f"Groq ({self.model})"

    def complete(self, prompt: str, system: str | None = None, timeout: float | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": 400}
        # gpt-oss / qwen / deepseek-r1-distill reasoning models on Groq burn
        # hidden "reasoning" tokens against the same TPM budget as the
        # visible completion. A short WhatsApp-message composer doesn't
        # need deep reasoning, so default to low effort to keep both token
        # usage and latency down. No-op on non-reasoning models.
        if any(tag in self.model for tag in ("gpt-oss", "qwen", "deepseek-r1")):
            body["reasoning_effort"] = os.environ.get("GROQ_REASONING_EFFORT", "low")
        data = _post_json(
            "https://api.groq.com/openai/v1/chat/completions",
            body,
            {"Authorization": f"Bearer {self.api_key}"},
            timeout=timeout or TIMEOUT_LLM,
        )
        return data["choices"][0]["message"]["content"]


class CerebrasProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = ""):
        self.api_key = api_key
        self.model = model or "llama-3.3-70b"

    def name(self) -> str:
        return f"Cerebras ({self.model})"

    def complete(self, prompt: str, system: str | None = None, timeout: float | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        data = _post_json(
            "https://api.cerebras.ai/v1/chat/completions",
            {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": 400},
            {"Authorization": f"Bearer {self.api_key}"},
            timeout=timeout or TIMEOUT_LLM,
        )
        return data["choices"][0]["message"]["content"]


def get_provider() -> LLMProvider | None:
    """Build a provider from env vars. Returns None if not configured —
    caller should fall back to the rule-based composer."""
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    model = os.environ.get("LLM_MODEL", "").strip()

    if not provider or provider == "none" or not api_key:
        return None

    try:
        if provider == "anthropic":
            return AnthropicProvider(api_key, model)
        if provider == "openai":
            return OpenAIProvider(api_key, model)
        if provider == "gemini":
            return GeminiProvider(api_key, model)
        if provider == "deepseek":
            return DeepSeekProvider(api_key, model)
        if provider == "groq":
            return GroqProvider(api_key, model)
        if provider == "cerebras":
            return CerebrasProvider(api_key, model)
    except Exception:
        return None
    return None
