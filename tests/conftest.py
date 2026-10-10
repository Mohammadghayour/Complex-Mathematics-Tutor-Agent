"""Shared pytest fixtures for the Complex Mathematics Tutor test suite.

Every fixture here is deliberately **offline**: the default configuration
carries no API key, so the agent uses the deterministic provider and the whole
suite runs without network access.  Tests that need a language model install a
``FakeProvider`` (see ``tests/fakes.py``) instead of calling a real service.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent

for candidate in (str(PROJECT_DIR), str(ROOT_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from complex_math_tutor.agent import TutorAgent, setup_logging  # noqa: E402
from complex_math_tutor.config import Config  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Remove every model related variable so tests never touch a real API."""
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "MATH_TUTOR_API_KEY",
        "MATH_TUTOR_PROVIDER",
        "MATH_TUTOR_MODEL",
        "MATH_TUTOR_VISION_MODEL",
        "MATH_TUTOR_MODEL_REVIEW",
        "MATH_TUTOR_IMAGES_DIR",
        "MATH_TUTOR_MAX_CLARIFICATIONS",
        "MATH_TUTOR_MAX_SOLVE_ATTEMPTS",
    ):
        monkeypatch.delenv(name, raising=False)
    # point the images directory at a temporary folder
    monkeypatch.setenv("MATH_TUTOR_IMAGES_DIR", str(tmp_path / "images"))


@pytest.fixture
def offline_config() -> Config:
    """A configuration that forces the deterministic offline provider."""
    return Config.from_env(provider="deterministic", api_key=None, debug=False)


@pytest.fixture
def offline_agent(offline_config: Config) -> TutorAgent:
    """An agent wired to the offline provider - no API key, no network."""
    setup_logging(debug=False)
    return TutorAgent(config=offline_config)


@pytest.fixture
def scripted_agent(offline_config: Config):
    """Factory building an offline agent driven by scripted clarification answers."""

    def _make(answers: list[str]) -> TutorAgent:
        iterator = iter(answers)

        def callback(_question: str) -> str:
            try:
                return next(iterator)
            except StopIteration:
                return ""

        return TutorAgent(config=offline_config, input_callback=callback)

    return _make


@pytest.fixture
def tiny_png(tmp_path: Path) -> Path:
    """A real, valid 1x1 PNG file used by the image tests."""
    import struct
    import zlib

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    header = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    raw = b"\x00\xff\x00\x00"
    data = header + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    path = tmp_path / "problem.png"
    path.write_bytes(data)
    return path
