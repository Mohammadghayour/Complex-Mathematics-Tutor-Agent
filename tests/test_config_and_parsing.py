"""Tests for configuration parsing and model response parsing.

These cover two real bugs found while wiring a custom OpenAI-compatible
endpoint (``OPENAI_BASE_URL`` with an inline ``#`` comment, and Persian section
labels the model phrased differently from the offline engine).  Both run
offline: no request is ever made.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from complex_math_tutor import heuristics as h
from complex_math_tutor.config import Config, _strip_inline_comment, parse_env_file
from complex_math_tutor.llm_provider import (
    coerce_answer,
    parse_json_object,
    parse_labelled_sections,
    split_numbered_steps,
)

# --------------------------------------------------------------------------- #
# .env parsing
# --------------------------------------------------------------------------- #

def test_inline_comment_is_stripped() -> None:
    """A trailing comment after a value must not become part of the value."""
    assert _strip_inline_comment("https://api.example.com/v1   # example") == "https://api.example.com/v1"
    assert _strip_inline_comment("gpt-4o-mini  # the model") == "gpt-4o-mini"


def test_hash_inside_url_is_kept() -> None:
    """A '#' that is not preceded by whitespace is part of the value."""
    assert _strip_inline_comment("https://host/v1#frag") == "https://host/v1#frag"


def test_hash_inside_quotes_is_kept() -> None:
    """A quoted value may contain '#' safely."""
    assert _strip_inline_comment('"a#b"') == '"a#b"'


def test_parse_env_file_handles_comments_and_quotes(tmp_path: Path) -> None:
    """Full-line comments, inline comments and quotes are all handled."""
    env = tmp_path / ".env"
    env.write_text(
        "\n".join(
            [
                "# a full line comment",
                "",
                "OPENAI_API_KEY=sk-test-123",
                "OPENAI_BASE_URL=https://api.gapgpt.app/v1   # example: local Ollama",
                'MATH_TUTOR_MODEL="gpt-4o-mini"',
                "MATH_TUTOR_MAX_CLARIFICATIONS=5",
            ]
        ),
        encoding="utf-8",
    )
    values = parse_env_file(env)
    assert values["OPENAI_API_KEY"] == "sk-test-123"
    assert values["OPENAI_BASE_URL"] == "https://api.gapgpt.app/v1"
    assert values["MATH_TUTOR_MODEL"] == "gpt-4o-mini"
    assert values["MATH_TUTOR_MAX_CLARIFICATIONS"] == "5"


def test_parse_env_file_missing_file_returns_empty(tmp_path: Path) -> None:
    """A missing .env is not an error."""
    assert parse_env_file(tmp_path / "nope.env") == {}


def test_custom_base_url_selects_the_compatible_provider(tmp_path: Path) -> None:
    """A key plus a custom base URL must select openai_compatible, not openai."""
    env = tmp_path / ".env"
    env.write_text(
        "OPENAI_API_KEY=sk-test\nOPENAI_BASE_URL=https://api.gapgpt.app/v1  # note\n",
        encoding="utf-8",
    )
    config = Config.from_env(env_file=env)
    assert config.api_key == "sk-test"
    assert config.base_url == "https://api.gapgpt.app/v1"
    assert config.resolved_provider == "openai_compatible"
    assert config.has_language_model is True


def test_no_key_means_offline_provider(tmp_path: Path) -> None:
    """Without a key the deterministic provider is selected."""
    env = tmp_path / ".env"
    env.write_text("MATH_TUTOR_PROVIDER=auto\n", encoding="utf-8")
    config = Config.from_env(env_file=env)
    assert config.resolved_provider == "deterministic"
    assert config.has_language_model is False


def test_env_var_overrides_the_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An exported variable wins over the .env file entry."""
    env = tmp_path / ".env"
    env.write_text("MATH_TUTOR_MODEL=from-file\n", encoding="utf-8")
    monkeypatch.setenv("MATH_TUTOR_MODEL", "from-env")
    assert Config.from_env(env_file=env).model == "from-env"


def test_vision_requires_a_key(tmp_path: Path) -> None:
    """has_vision stays False when only a base URL is configured."""
    env = tmp_path / ".env"
    env.write_text("OPENAI_BASE_URL=https://api.example.com/v1\n", encoding="utf-8")
    config = Config.from_env(env_file=env)
    assert config.has_language_model is False
    assert config.has_vision is False


# --------------------------------------------------------------------------- #
# Model response parsing
# --------------------------------------------------------------------------- #

def test_persian_section_labels_are_recognised() -> None:
    """Alternative Persian headings must map to the canonical sections."""
    text = "\n".join(
        [
            "تعریف: مشتق نرخ تغییر است.",
            "شعور: مثل سرعت خودرو در یک لحظه.",
            "مثال: مشتق x^2 برابر 2x است.",
            "نکات کلیدی: قاعده توان، مشتق ثابت صفر.",
        ]
    )
    sections = parse_labelled_sections(text)
    assert sections["definition"] == "مشتق نرخ تغییر است."
    assert "سرعت خودرو" in sections["intuition"]
    assert "2x" in sections["example"]
    assert "قاعده توان" in sections["notes"]


def test_markdown_headings_and_numbering_are_stripped() -> None:
    """Bold, hashed and numbered headings all parse."""
    text = "\n".join(
        [
            "**تعریف ساده:** مشتق شیب خط مماس است.",
            "## توضیح شهودی:",
            "مثل سرعت لحظه‌ای.",
            "۳) مثال: f(x) = x^2",
            "- نکات مهم: قاعده توان.",
        ]
    )
    sections = parse_labelled_sections(text)
    assert "شیب خط مماس" in sections["definition"]
    assert "سرعت لحظه‌ای" in sections["intuition"]
    assert "x^2" in sections["example"]
    assert "قاعده توان" in sections["notes"]


def test_unknown_label_is_not_merged_into_the_previous_section() -> None:
    """An unrecognised heading starts its own bucket instead of polluting one."""
    text = "\n".join(
        [
            "تعریف: مشتق نرخ تغییر است.",
            "خلاصه: این یک برچسب ناشناخته است.",
        ]
    )
    sections = parse_labelled_sections(text)
    assert sections["definition"] == "مشتق نرخ تغییر است."
    assert "خلاصه" not in sections["definition"]
    assert any(key.startswith("other:") for key in sections)


def test_engineering_use_heading_maps_to_notes() -> None:
    """A fuzzy Persian heading such as 'کاربردها' still lands in notes."""
    sections = parse_labelled_sections("کاربردهای مهندسی: در تحلیل سازه.")
    assert "تحلیل سازه" in sections.get("notes", "")


def test_solution_sections_parse() -> None:
    """The solver's Analysis/Method/Steps/Final answer sections parse."""
    text = "\n".join(
        [
            "Analysis: we have x = 2 and want 3x^2 + 1",
            "Method: substitution",
            "Steps:",
            "1. substitute x = 2",
            "2. compute 3*4 + 1 = 13",
            "Final answer: 13",
        ]
    )
    sections = parse_labelled_sections(text)
    assert "x = 2" in sections["analysis"]
    assert sections["method"] == "substitution"
    assert sections["answer"] == "13"
    steps = split_numbered_steps(sections["steps"])
    assert len(steps) == 2
    assert "substitute" in steps[0]


@pytest.mark.parametrize(
    "raw,expected",
    [("13", 13.0), ("-2.5", -2.5), ("1/2", 0.5), ("3,000", 3000.0), ("2x + 3", "2x + 3")],
)
def test_coerce_answer(raw: str, expected: object) -> None:
    """Textual answers become numbers only when that is unambiguous."""
    result = coerce_answer(raw)
    if isinstance(expected, float):
        assert result == pytest.approx(expected)
    else:
        assert result == expected


def test_parse_json_object_tolerates_prose() -> None:
    """JSON embedded in prose is still extracted."""
    assert parse_json_object('Sure!\n{"request_type": "conceptual"}\nThanks') == {
        "request_type": "conceptual"
    }
    assert parse_json_object("no json here") == {}


def test_clarification_prompt_forces_the_student_language() -> None:
    """The prompt must tell the model which language to answer in."""
    from complex_math_tutor import prompts

    fa = prompts.clarification_prompt("سلام", "no_math", None, "fa")
    assert "PERSIAN" in fa
    en = prompts.clarification_prompt("hi", "no_math", None, "en")
    assert "English" in en


def test_topic_is_normalised_onto_a_canonical_key() -> None:
    """Free-text topics from the model become canonical keys for the header."""
    from complex_math_tutor.llm_provider import OpenAICompatibleProvider as Provider

    assert Provider._normalise_topic("integral", "unknown") == "integral"
    assert Provider._normalise_topic("انتگرال", "unknown") == "integral"
    assert Provider._normalise_topic("calculus - integrals", "unknown") == "integral"
    assert Provider._normalise_topic("derivatives", "unknown") == "derivative"
    assert Provider._normalise_topic("", "unknown") == "unknown"
    # an unknown topic falls back to whatever the offline detector found
    assert Provider._normalise_topic("", "algebra") == "algebra"


def test_model_text_wins_over_the_offline_fallback() -> None:
    """A parsed model section must not be replaced by offline boilerplate."""
    model_text = "\n".join(
        [
            "تعریف: تعریف مدل.",
            "شعور: شهود مدل.",
            "مثال: مثال مدل.",
            "نکات کلیدی: نکته مدل.",
        ]
    )
    sections = parse_labelled_sections(model_text)
    fallback = h.explain_concept("derivative", "مشتق چیست", "beginner", "fa")
    merged = {key: sections.get(key) or fallback[key] for key in ("definition", "intuition", "example", "notes")}
    assert merged["definition"] == "تعریف مدل."
    assert merged["notes"] == "نکته مدل."
    assert fallback["definition"] not in merged["definition"]
