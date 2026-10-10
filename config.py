"""Runtime configuration for the Complex Mathematics Tutor Agent.

Configuration is read from environment variables and (optionally) from a
``.env`` file placed next to this module.  **No API key is ever hard-coded** -
see ``.env.example`` for the supported variables.

Resolution order for every setting:

1. explicit value passed by the caller (CLI argument, test, graph config),
2. environment variable,
3. ``.env`` file entry,
4. built-in default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "images"

SUPPORTED_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")
MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB
IMAGE_MAGIC_NUMBERS: Mapping[str, tuple[bytes, ...]] = {
    ".png": (b"\x89PNG\r\n\x1a\n",),
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
    ".gif": (b"GIF87a", b"GIF89a"),
    ".bmp": (b"BM",),
    ".webp": (b"RIFF",),
}

#: Provider names understood by :mod:`complex_math_tutor.llm_provider`.
KNOWN_PROVIDERS = ("auto", "openai", "openai_compatible", "ollama", "deterministic")

#: Providers that can accept an image payload.
VISION_PROVIDERS = ("openai", "openai_compatible")


def _strip_inline_comment(value: str) -> str:
    """Remove a trailing ``# comment`` from a .env value.

    A ``#`` only starts a comment when it is preceded by whitespace and is not
    inside quotes, so ``https://host/v1#frag`` and ``"a#b"`` survive intact.
    """
    if "#" not in value:
        return value
    quote: str | None = None
    for index, char in enumerate(value):
        if quote:
            if char == quote:
                quote = None
            continue
        if char in "\"'":
            quote = char
            continue
        if char == "#" and index > 0 and value[index - 1].isspace():
            return value[:index].strip()
    return value


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse a minimal ``KEY=VALUE`` .env file.

    Full-line comments (``# ...``) are ignored and inline comments
    (``KEY=value  # note``) are stripped, which matters because the example
    file documents options next to the value.  Surrounding quotes are removed.
    """
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = _strip_inline_comment(value).strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


@dataclass(slots=True)
class Config:
    """Immutable-ish configuration object shared by the CLI and the graph."""

    provider: str = "auto"
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.2
    timeout: float = 60.0
    max_tokens: int = 1200
    vision_model: str | None = None
    max_clarification_rounds: int = 3
    max_solve_attempts: int = 2
    debug: bool = False
    allow_model_review: bool = True
    images_dir: Path = field(default_factory=lambda: DEFAULT_IMAGES_DIR)
    env_file: Path = field(default_factory=lambda: DEFAULT_ENV_FILE)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_env(
        cls,
        env_file: str | os.PathLike[str] | None = None,
        **overrides: Any,
    ) -> "Config":
        """Build a config from the environment plus explicit overrides."""
        path = Path(env_file).expanduser() if env_file else DEFAULT_ENV_FILE
        file_values = parse_env_file(path)

        def read(name: str, default: str | None = None) -> str | None:
            if name in os.environ and os.environ[name] != "":
                return os.environ[name]
            return file_values.get(name, default)

        def read_bool(name: str, default: bool) -> bool:
            raw = read(name)
            if raw is None:
                return default
            return raw.strip().lower() in {"1", "true", "yes", "on"}

        def read_int(name: str, default: int) -> int:
            raw = read(name)
            try:
                return int(raw) if raw is not None else default
            except (TypeError, ValueError):
                return default

        def read_float(name: str, default: float) -> float:
            raw = read(name)
            try:
                return float(raw) if raw is not None else default
            except (TypeError, ValueError):
                return default

        config = cls(
            provider=(read("MATH_TUTOR_PROVIDER", "auto") or "auto").strip().lower(),
            model=read("MATH_TUTOR_MODEL", "gpt-4o-mini") or "gpt-4o-mini",
            api_key=read("OPENAI_API_KEY") or read("MATH_TUTOR_API_KEY"),
            base_url=read("OPENAI_BASE_URL") or read("MATH_TUTOR_BASE_URL"),
            temperature=read_float("MATH_TUTOR_TEMPERATURE", 0.2),
            timeout=read_float("MATH_TUTOR_TIMEOUT", 60.0),
            max_tokens=read_int("MATH_TUTOR_MAX_TOKENS", 1200),
            vision_model=read("MATH_TUTOR_VISION_MODEL"),
            max_clarification_rounds=read_int("MATH_TUTOR_MAX_CLARIFICATIONS", 3),
            max_solve_attempts=read_int("MATH_TUTOR_MAX_SOLVE_ATTEMPTS", 2),
            allow_model_review=read_bool("MATH_TUTOR_MODEL_REVIEW", True),
            images_dir=Path(read("MATH_TUTOR_IMAGES_DIR", str(DEFAULT_IMAGES_DIR)) or DEFAULT_IMAGES_DIR),
            env_file=path,
        )
        clean = {k: v for k, v in overrides.items() if v is not None}
        return config.with_overrides(**clean) if clean else config

    def with_overrides(self, **overrides: Any) -> "Config":
        """Return a copy of this config with the given fields replaced."""
        unknown = set(overrides) - {f for f in self.__dataclass_fields__}
        if unknown:
            raise ValueError(f"unknown configuration field(s): {sorted(unknown)}")
        return replace(self, **overrides)

    # ------------------------------------------------------------------ #
    # Derived properties
    # ------------------------------------------------------------------ #
    @property
    def resolved_vision_model(self) -> str:
        """Model used for image analysis (falls back to the main model)."""
        return self.vision_model or self.model

    @property
    def resolved_provider(self) -> str:
        """Resolve ``auto`` into a concrete provider name."""
        if self.provider != "auto":
            return self.provider
        if self.api_key:
            return "openai" if not self.base_url else "openai_compatible"
        return "deterministic"

    @property
    def has_language_model(self) -> bool:
        """True when a real model is reachable with the current settings."""
        return self.resolved_provider != "deterministic"

    @property
    def has_vision(self) -> bool:
        """True when the configured provider can analyse images."""
        return self.resolved_provider in VISION_PROVIDERS and bool(self.api_key)

    def describe(self) -> str:
        """One-line human readable description used by the CLI banner."""
        if self.has_language_model:
            target = f"{self.resolved_provider}:{self.model}"
        else:
            target = "deterministic (offline fallback, no API key)"
        vision = self.resolved_vision_model if self.has_vision else "unavailable"
        return f"provider={target} | vision={vision}"
