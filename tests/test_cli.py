"""Tests for the CLI: argument parsing, config merging and the interactive loop.

The CLI is driven through :func:`complex_math_tutor.cli.main` with patched
``input``/``stdin`` so the tests never block and never need a real model.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

from complex_math_tutor import cli
from complex_math_tutor.agent import TutorAgent
from complex_math_tutor.config import Config
from complex_math_tutor.llm_provider import DeterministicProvider
from fakes import FakeProvider


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #

def test_parser_accepts_the_documented_flags() -> None:
    """--model, --provider, --image and --debug all exist."""
    parser = cli.build_parser()
    args = parser.parse_args(
        ["--model", "gpt-4o-mini", "--provider", "deterministic", "--image", "x.png", "--debug"]
    )
    assert args.model == "gpt-4o-mini"
    assert args.provider == "deterministic"
    assert args.image == "x.png"
    assert args.debug is True


def test_parser_rejects_an_unknown_provider() -> None:
    """Only known providers are accepted."""
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--provider", "not-a-provider"])


def test_config_from_args_merges_cli_and_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI flags win over environment defaults."""
    monkeypatch.setenv("MATH_TUTOR_MODEL", "from-env")
    parser = cli.build_parser()
    args = parser.parse_args(["--model", "from-cli", "--no-model-review"])
    config = cli.config_from_args(args)
    assert config.model == "from-cli"
    assert config.allow_model_review is False


def test_config_defaults_to_the_offline_provider_without_a_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """With no API key the CLI silently falls back to the offline engine.

    ``--env`` points at a throwaway file so the test is isolated from any real
    ``.env`` sitting in the project directory.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    empty_env = tmp_path / "empty.env"
    empty_env.write_text("# no key here\n", encoding="utf-8")

    parser = cli.build_parser()
    config = cli.config_from_args(parser.parse_args(["--env", str(empty_env)]))
    assert config.resolved_provider == "deterministic"
    assert config.has_language_model is False


def test_cli_uses_a_configured_key_and_base_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A key plus a custom base URL really switches the CLI to the model."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    env = tmp_path / "custom.env"
    env.write_text(
        "OPENAI_API_KEY=sk-test-abc\n"
        "OPENAI_BASE_URL=https://api.example.com/v1  # my gateway\n"
        "MATH_TUTOR_MODEL=gpt-4o-mini\n",
        encoding="utf-8",
    )

    parser = cli.build_parser()
    config = cli.config_from_args(parser.parse_args(["--env", str(env)]))
    assert config.api_key == "sk-test-abc"
    assert config.base_url == "https://api.example.com/v1"
    assert config.resolved_provider == "openai_compatible"
    assert config.has_language_model is True
    assert config.has_vision is True


# --------------------------------------------------------------------------- #
# Level command
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "value,expected",
    [("beginner", "beginner"), ("ADVANCED", "advanced"), ("auto", None)],
)
def test_resolve_level_argument(value: str, expected: str | None) -> None:
    """Valid level arguments are accepted and 'auto' clears the pin."""
    assert cli.resolve_level_argument(value, None) == expected


def test_resolve_level_argument_rejects_garbage(capsys: pytest.CaptureFixture[str]) -> None:
    """An invalid level keeps the previous value and prints a message."""
    assert cli.resolve_level_argument("wizard", "beginner") == "beginner"
    assert "سطح نامعتبر" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def test_render_turn_prints_the_header_and_answer(offline_agent: TutorAgent, capsys: pytest.CaptureFixture[str]) -> None:
    """A completed turn is rendered with type/topic/level and the answer."""
    state = offline_agent.run("مشتق چیست؟")
    cli.render_turn(offline_agent, state)
    out = capsys.readouterr().out
    assert "نوع درخواست" in out
    assert "موضوع" in out
    assert "سطح تقریبی" in out
    assert "تعریف ساده" in out


def test_render_turn_debug_shows_the_graph_path(offline_agent: TutorAgent, capsys: pytest.CaptureFixture[str]) -> None:
    """--debug adds the executed graph path to the output."""
    state = offline_agent.run("مشتق چیست؟")
    cli.render_turn(offline_agent, state, debug=True)
    out = capsys.readouterr().out
    assert "مسیر اجرای گراف" in out
    assert "explain_concept" in out


# --------------------------------------------------------------------------- #
# Interactive loop
# --------------------------------------------------------------------------- #

def _make_cli(agent: TutorAgent, inputs: list[str], debug: bool = False) -> cli.TutorCLI:
    """Build a CLI whose reader replays *inputs* and then raises EOF."""
    instance = cli.TutorCLI(agent, debug=debug)
    iterator = iter(inputs)

    def reader(_prompt: str = "") -> str:
        try:
            return next(iterator)
        except StopIteration:
            raise EOFError() from None

    instance.reader = reader  # type: ignore[assignment]
    return instance


def test_cli_exits_on_exit_command(offline_config: Config, capsys: pytest.CaptureFixture[str]) -> None:
    """'exit' terminates the loop cleanly."""
    agent = TutorAgent(config=offline_config)
    instance = _make_cli(agent, ["exit"])
    assert instance.run() == 0
    assert "Goodbye" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["quit", ":q", ":quit"])
def test_cli_exit_aliases(offline_config: Config, command: str) -> None:
    """quit and the colon aliases also terminate the loop."""
    instance = _make_cli(TutorAgent(config=offline_config), [command])
    assert instance.run() == 0


def test_cli_handles_eof_gracefully(offline_config: Config) -> None:
    """Ctrl+D (EOFError) exits with code 0 instead of crashing."""
    agent = TutorAgent(config=offline_config)
    instance = cli.TutorCLI(agent)
    instance.reader = lambda _prompt="": (_ for _ in ()).throw(EOFError())  # type: ignore[assignment]
    assert instance.run() == 0


def test_cli_answers_a_question(offline_config: Config, capsys: pytest.CaptureFixture[str]) -> None:
    """A typed question is answered through the graph."""
    instance = _make_cli(
        TutorAgent(config=offline_config),
        ["اگر x=2 باشد، مقدار 3x^2 + 1 چقدر است؟", "exit"],
    )
    assert instance.run() == 0
    out = capsys.readouterr().out
    assert "13" in out
    assert "نوع درخواست" in out


def test_cli_help_and_reset(offline_config: Config, capsys: pytest.CaptureFixture[str]) -> None:
    """help prints the command list and reset clears the context."""
    instance = _make_cli(TutorAgent(config=offline_config), ["help", "مشتق چیست؟", "reset", "exit"])
    assert instance.run() == 0
    out = capsys.readouterr().out
    assert "Commands" in out
    assert "زمینه گفتگو پاک شد" in out
    assert instance.state is None


def test_cli_level_command_pins_the_level(offline_config: Config, capsys: pytest.CaptureFixture[str]) -> None:
    """`level advanced` is applied to the next question."""
    instance = _make_cli(TutorAgent(config=offline_config), ["level advanced", "exit"])
    assert instance.run() == 0
    assert instance.pinned_level == "advanced"
    assert "سطح فعلی: advanced" in capsys.readouterr().out


def test_cli_unknown_command_does_not_crash(offline_config: Config, capsys: pytest.CaptureFixture[str]) -> None:
    """An unknown command prints a hint and the loop continues."""
    instance = _make_cli(TutorAgent(config=offline_config), [":frobnicate", "exit"])
    assert instance.run() == 0
    assert "دستور ناشناخته" in capsys.readouterr().out


def test_cli_scripted_clarification_round(offline_config: Config, capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI collects a follow-up answer when the agent asks for one."""
    instance = _make_cli(
        TutorAgent(config=offline_config),
        [
            "سلام",
            "اگر x=2 باشد، مقدار 3x^2 + 1 چقدر است؟",
            "exit",
        ],
    )
    assert instance.run() == 0
    out = capsys.readouterr().out
    assert "Agent:" in out  # the follow-up question was shown
    assert "13" in out


def test_cli_once_mode_answers_and_exits(offline_config: Config, capsys: pytest.CaptureFixture[str]) -> None:
    """--once answers a single query without entering the loop."""
    instance = _make_cli(TutorAgent(config=offline_config), [])
    assert instance.run(once="مشتق چیست؟") == 0
    assert "تعریف" in capsys.readouterr().out


def test_cli_reports_a_model_error_without_crashing(offline_config: Config, capsys: pytest.CaptureFixture[str]) -> None:
    """A provider exception is turned into a readable message."""
    from complex_math_tutor.llm_provider import LLMError

    class ExplodingProvider(DeterministicProvider):
        def classify(self, *args, **kwargs):  # type: ignore[override]
            raise LLMError("boom")

    agent = TutorAgent(config=offline_config, provider=ExplodingProvider())
    instance = _make_cli(agent, ["مشتق چیست؟", "exit"])
    assert instance.run() == 0
    out = capsys.readouterr().out
    assert "classify" not in out  # not the raw provider error
    assert "13" in out or "تعریف" in out or "هشدار" in out


def test_main_runs_with_mocked_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """main() works end to end in mock mode (no API key, no network)."""
    monkeypatch.setenv("MATH_TUTOR_PROVIDER", "deterministic")
    monkeypatch.setenv("MATH_TUTOR_IMAGES_DIR", str(tmp_path / "images"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(sys, "stdin", io.StringIO("اگر x=2 باشد، مقدار 3x^2 + 1 چقدر است؟\nexit\n"))
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert "Complex Mathematics Tutor Agent" in out
    assert "13" in out


def test_main_once_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """main() honours --once with the offline provider."""
    monkeypatch.setenv("MATH_TUTOR_PROVIDER", "deterministic")
    monkeypatch.setenv("MATH_TUTOR_IMAGES_DIR", str(tmp_path / "images"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert cli.main(["--once", "2x + 3 = 7"]) == 0
    assert "2" in capsys.readouterr().out


def test_cli_uses_a_fake_provider_end_to_end(
    offline_config: Config,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A scripted provider drives the CLI without any network access."""
    provider = FakeProvider(
        classify_result={"request_type": "conceptual", "subject_or_topic": "derivative", "is_clear": True}
    )
    instance = _make_cli(TutorAgent(config=offline_config, provider=provider), ["مشتق چیست؟", "exit"])
    assert instance.run() == 0
    out = capsys.readouterr().out
    assert "تعریف آزمایشی" in out
    assert "classify" in provider.calls
    assert "explain" in provider.calls
