"""Scripted fake providers used by the test suite.

A :class:`FakeProvider` implements the same protocol as the real providers but
returns canned answers, so the graph can be exercised deterministically and
without any API key.  It is also the reference example for anyone adding a new
provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from complex_math_tutor import heuristics as h
from complex_math_tutor.llm_provider import ImageNotSupportedError


@dataclass
class FakeProvider:
    """Deterministic, fully scripted provider for tests."""

    name: str = "fake"
    supports_vision: bool = False
    classify_result: dict[str, Any] = field(default_factory=dict)
    clarify_text: str = "لطفاً صورت مسئله را کامل بنویسید."
    concept_sections: dict[str, str] = field(
        default_factory=lambda: {
            "definition": "تعریف آزمایشی",
            "intuition": "شهود آزمایشی",
            "example": "مثال آزمایشی",
            "notes": "نکته آزمایشی",
        }
    )
    solve_result: dict[str, Any] = field(
        default_factory=lambda: {
            "steps": ["1. جایگذاری مقادیر", "2. ساده‌سازی", "3. محاسبه نهایی"],
            "answer": 13.0,
            "method": "روش آزمایشی",
        }
    )
    review_result: dict[str, Any] = field(
        default_factory=lambda: {"is_correct": True, "notes": ["بررسی آزمایشی"], "corrected_answer": 13.0}
    )
    image_text: str = "x^2 + 3x = 0"

    #: recorded calls, useful for assertions
    calls: list[str] = field(default_factory=list)

    # -- protocol ------------------------------------------------------- #
    def classify(self, query: str, history: Sequence[Mapping[str, str]] | None = None) -> dict[str, Any]:
        self.calls.append("classify")
        base = {
            "request_type": "conceptual",
            "subject_or_topic": "derivative",
            "user_level": "beginner",
            "problem_statement": "",
            "language": h.detect_language(query),
            "is_clear": True,
            "missing_information": "none",
            "clarification_question": "",
            "confidence": 1.0,
            "provider": self.name,
        }
        base.update(self.classify_result)
        return base

    def clarify(self, query: str, missing: str, history: Sequence[Mapping[str, str]] | None = None) -> str:
        self.calls.append("clarify")
        return self.clarify_text

    def explain(
        self,
        query: str,
        topic: str,
        level: str,
        language: str,
        history: Sequence[Mapping[str, str]] | None = None,
    ) -> dict[str, str]:
        self.calls.append("explain")
        return dict(self.concept_sections)

    def solve(
        self,
        query: str,
        problem_statement: str,
        level: str,
        language: str,
        previous_attempt: Mapping[str, Any] | None = None,
        feedback: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        self.calls.append("solve")
        result = dict(self.solve_result)
        result.setdefault("problem_statement", problem_statement or query)
        result.setdefault("given", {})
        result.setdefault("unknown", "final value")
        result.setdefault("expression", h.extract_math_expression(problem_statement or query))
        result.setdefault("assignments", {})
        result.setdefault("provider", self.name)
        return result

    def review(
        self,
        problem_statement: str,
        steps: Sequence[str],
        candidate_answer: Any,
        kind: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append("review")
        return dict(self.review_result)

    def inspect_image(self, image_path: str | Path, question: str | None = None) -> str:
        self.calls.append("inspect_image")
        if not self.supports_vision:
            raise ImageNotSupportedError("fake provider has no vision")
        return self.image_text


@dataclass
class FailingProvider(FakeProvider):
    """Provider whose ``solve`` raises, used to test graceful degradation."""

    name: str = "failing"

    def solve(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        from complex_math_tutor.llm_provider import LLMError

        self.calls.append("solve")
        raise LLMError("simulated model outage")
