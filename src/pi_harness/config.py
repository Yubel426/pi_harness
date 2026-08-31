"""Configuration for the terminal harness."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def normalize_base_url(value: str) -> str:
    """Normalize an OpenAI-compatible base URL and add ``/v1`` when absent."""

    raw = value.strip().rstrip("/")
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid API base URL: {value!r}")

    path = parsed.path.rstrip("/")
    if not path.endswith("/v1") and path != "v1":
        path = f"{path}/v1" if path else "/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, ""))


@dataclass(frozen=True, slots=True)
class HarnessConfig:
    """Runtime settings; Pi resolves provider credentials separately."""

    api_key: str | None = None
    base_url: str | None = None
    provider: str = "openai"
    model: str = DEFAULT_MODEL
    reasoning_effort: str | None = "medium"
    max_output_tokens: int | None = None
    request_timeout: float = 600.0
    max_tool_rounds: int = 100
    cwd: Path = field(default_factory=Path.cwd)

    def __post_init__(self) -> None:
        if self.api_key is not None and not self.api_key.strip():
            raise ValueError("API key cannot be empty")
        if self.base_url and self.provider == "openai":
            object.__setattr__(self, "base_url", normalize_base_url(self.base_url))
        if not self.provider.strip():
            raise ValueError("Provider cannot be empty")
        object.__setattr__(self, "cwd", self.cwd.expanduser().resolve())
        if not self.cwd.is_dir():
            raise ValueError(f"Working directory does not exist: {self.cwd}")
        if self.reasoning_effort not in (*REASONING_EFFORTS, None):
            raise ValueError(f"Invalid reasoning effort: {self.reasoning_effort}")
        if self.max_output_tokens is not None and self.max_output_tokens < 16:
            raise ValueError("max_output_tokens must be at least 16")
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        if self.max_tool_rounds <= 0:
            raise ValueError("max_tool_rounds must be positive")

    @classmethod
    def from_environment(
        cls,
        *,
        api_key: str | None = None,
        provider: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        max_output_tokens: int | None = None,
        request_timeout: float = 600.0,
        max_tool_rounds: int = 100,
        cwd: str | Path | None = None,
    ) -> HarnessConfig:
        resolved_provider = provider or os.environ.get("PI_PROVIDER") or "openai"
        # Native provider env vars are resolved by Pi. Do not accidentally use
        # an OpenAI key for Anthropic, Codex OAuth, or another provider.
        resolved_key = api_key or os.environ.get("PI_API_KEY")

        env_effort = os.environ.get("PI_REASONING_EFFORT")
        effort = (
            reasoning_effort
            if reasoning_effort is not None
            else (env_effort or "medium")
        )
        if effort == "off":
            effort = None

        return cls(
            api_key=resolved_key,
            provider=resolved_provider,
            base_url=base_url
            or os.environ.get("PI_BASE_URL")
            or (
                os.environ.get("OPENAI_BASE_URL")
                if resolved_provider == "openai"
                else None
            ),
            model=model or os.environ.get("PI_MODEL") or DEFAULT_MODEL,
            reasoning_effort=effort,
            max_output_tokens=max_output_tokens,
            request_timeout=request_timeout,
            max_tool_rounds=max_tool_rounds,
            cwd=Path(cwd) if cwd is not None else Path.cwd(),
        )
