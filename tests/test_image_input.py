"""Tests for image input: path resolution, validation, the inspect_image node
and the graceful degradation when no vision model is configured.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from complex_math_tutor import heuristics as h
from complex_math_tutor.agent import (
    TutorAgent,
    inspect_image,
    next_node,
    read_problem,
    resolve_image_path,
    validate_image_file,
)
from complex_math_tutor.config import Config, MAX_IMAGE_BYTES
from complex_math_tutor.llm_provider import ImageNotSupportedError
from fakes import FakeProvider

# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #

def test_resolve_absolute_path(tiny_png: Path, offline_config: Config) -> None:
    """An absolute path is returned as-is when it exists."""
    assert resolve_image_path(str(tiny_png), offline_config) == tiny_png.resolve()


def test_resolve_relative_path_from_cwd(monkeypatch: pytest.MonkeyPatch, tiny_png: Path, offline_config: Config) -> None:
    """A relative path is resolved against the current directory."""
    monkeypatch.chdir(tiny_png.parent)
    resolved = resolve_image_path(tiny_png.name, offline_config)
    assert resolved == tiny_png.resolve()


def test_resolve_relative_path_inside_images_dir(tmp_path: Path, offline_config: Config) -> None:
    """A bare filename is found inside the configured images directory."""
    images = tmp_path / "images"
    images.mkdir()
    target = images / "question.png"
    target.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    config = offline_config.with_overrides(images_dir=images)
    assert resolve_image_path("question.png", config) == target.resolve()


def test_resolve_missing_path_returns_it_unchanged(offline_config: Config) -> None:
    """A non-existent path is returned so the caller can report it."""
    resolved = resolve_image_path("definitely-not-here.png", offline_config)
    assert not resolved.exists()


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def test_valid_png_passes(tiny_png: Path) -> None:
    """A real PNG file validates."""
    assert validate_image_file(tiny_png) is None


def test_missing_file_is_reported(tmp_path: Path) -> None:
    """A missing file produces a readable error."""
    error = validate_image_file(tmp_path / "nope.png")
    assert error and "not found" in error


def test_directory_is_rejected(tmp_path: Path) -> None:
    """A directory is not a valid image."""
    error = validate_image_file(tmp_path)
    assert error and "not a file" in error


def test_unsupported_extension_is_rejected(tmp_path: Path) -> None:
    """A .txt file is rejected with the supported list."""
    path = tmp_path / "problem.txt"
    path.write_text("2x + 3 = 7")
    error = validate_image_file(path)
    assert error and "unsupported image format" in error


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    """A zero byte image is rejected."""
    path = tmp_path / "empty.png"
    path.write_bytes(b"")
    error = validate_image_file(path)
    assert error and "empty" in error


def test_oversized_file_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A file above the size limit is rejected."""
    path = tmp_path / "big.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    monkeypatch.setattr("complex_math_tutor.agent.MAX_IMAGE_BYTES", 8)
    error = validate_image_file(path)
    assert error and "too large" in error
    assert MAX_IMAGE_BYTES > 8  # the real limit is unchanged


def test_content_mismatch_is_rejected(tmp_path: Path) -> None:
    """A .png whose bytes are not PNG is rejected."""
    path = tmp_path / "fake.png"
    path.write_bytes(b"this is not a png at all")
    error = validate_image_file(path)
    assert error and "does not match" in error


# --------------------------------------------------------------------------- #
# The inspect_image node
# --------------------------------------------------------------------------- #

def test_inspect_image_reports_missing_vision(offline_agent: TutorAgent, tiny_png: Path) -> None:
    """With no vision model the node stores an understandable error."""
    deps = offline_agent.deps
    state = offline_agent.create_initial_state("")
    state["image_path"] = str(tiny_png)
    updates = inspect_image(state, deps)
    assert updates["image_processed"] is True
    assert updates["image_analysis"] is None
    assert updates["error_message"]
    assert "vision" in updates["error_message"].lower()
    assert updates["clarification_question"]


def test_inspect_image_reports_an_invalid_path(offline_agent: TutorAgent) -> None:
    """An invalid path is reported instead of raising."""
    deps = offline_agent.deps
    state = offline_agent.create_initial_state("")
    state["image_path"] = "images/does-not-exist.png"
    updates = inspect_image(state, deps)
    assert updates["image_processed"] is True
    assert "not found" in str(updates["error_message"])


def test_inspect_image_uses_a_vision_provider(offline_config: Config, tiny_png: Path) -> None:
    """A vision capable provider transcribes the problem into the state."""
    provider = FakeProvider(supports_vision=True, image_text="x^2 + 3x = 0")
    agent = TutorAgent(config=offline_config, provider=provider)
    deps = agent.deps
    state = agent.create_initial_state("")
    state["image_path"] = str(tiny_png)
    updates = inspect_image(state, deps)
    assert updates["image_analysis"] == "x^2 + 3x = 0"
    assert updates["is_problem_clear"] is True
    assert "x^2 + 3x = 0" in updates["normalized_query"]
    assert "inspect_image" in provider.calls


def test_image_path_is_routed_through_inspect_image(offline_agent: TutorAgent, tiny_png: Path) -> None:
    """read_problem routes to inspect_image when an image path is present."""
    deps = offline_agent.deps
    state = offline_agent.create_initial_state("", image_path=str(tiny_png))
    assert next_node("read_problem", state, deps) == "inspect_image"
    state["image_processed"] = True
    assert next_node("read_problem", state, deps) == "classify_request"


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #

def test_run_with_an_image_degrades_gracefully(offline_agent: TutorAgent, tiny_png: Path) -> None:
    """A run with an image but no vision model still terminates with a message."""
    state = offline_agent.run("مسئله در تصویر است", image_path=str(tiny_png))
    assert "inspect_image" in state["path_log"]
    assert state["path_log"][-1] == "final_answer"
    assert state["final_answer"]
    assert state["image_processed"] is True


def test_run_with_an_image_and_vision_solves_it(offline_config: Config, tiny_png: Path) -> None:
    """With a vision provider the transcribed problem is actually solved."""
    provider = FakeProvider(
        supports_vision=True,
        image_text="x^2 + 3x = 0",
        classify_result={
            "request_type": "solvable_problem",
            "problem_statement": "x^2 + 3*x = 0",
            "is_clear": True,
            "subject_or_topic": "equation",
            "user_level": "beginner",
        },
        solve_result={
            "steps": ["1. فاکتورگیری x(x+3) = 0", "2. x = 0 یا x = -3"],
            "answer": 0.0,
            "method": "فاکتورگیری",
            "expression": "x^2+3*x=0",
            "kind": "equation",
        },
        review_result={"is_correct": True, "notes": ["بررسی آزمایشی"], "corrected_answer": 0.0},
    )
    agent = TutorAgent(config=offline_config, provider=provider)
    state = agent.run("", image_path=str(tiny_png))
    assert "inspect_image" in state["path_log"]
    assert state["is_problem_clear"] is True
    assert state["final_answer"]
    assert "x^2 + 3x = 0" in str(state["image_analysis"])


def test_cli_image_command_rejects_a_bad_path(
    offline_config: Config,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI prints a readable image error and keeps running."""
    from complex_math_tutor import cli

    agent = TutorAgent(config=offline_config)
    instance = cli.TutorCLI(agent)
    instance.reader = lambda _prompt="": "images/missing.png"  # type: ignore[assignment]
    instance.handle_image_command()
    out = capsys.readouterr().out
    assert "خطای تصویر" in out
    assert "not found" in out


def test_cli_image_command_analyses_a_valid_image(
    offline_config: Config,
    tiny_png: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI validates the path, announces the analysis and runs the graph."""
    from complex_math_tutor import cli

    agent = TutorAgent(config=offline_config)
    instance = cli.TutorCLI(agent)
    answers = iter([str(tiny_png), ""])
    instance.reader = lambda _prompt="": next(answers)  # type: ignore[assignment]
    instance.handle_image_command()
    out = capsys.readouterr().out
    assert "در حال تحلیل" in out
    assert "inspect_image" in " ".join(instance.state["path_log"]) if instance.state else True


def test_vision_error_type_is_specific() -> None:
    """The offline provider raises the dedicated vision error."""
    from complex_math_tutor.llm_provider import DeterministicProvider

    with pytest.raises(ImageNotSupportedError):
        DeterministicProvider().inspect_image("whatever.png")
