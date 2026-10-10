"""LLM provider abstraction for the Complex Mathematics Tutor Agent.

The graph never talks to a vendor SDK directly - it talks to an
:class:`LLMProvider`.  Two implementations ship with the project:

``DeterministicProvider``
    A fully offline "model" built on :mod:`complex_math_tutor.heuristics`.
    It classifies, clarifies, explains and solves real mathematics, which
    means the graph, the CLI and the tests all work with **no API key**.

``OpenAICompatibleProvider``
    Talks to any OpenAI-compatible ``/chat/completions`` endpoint (OpenAI,
    Groq, Together, OpenRouter, a local vLLM/Ollama gateway, ...) using only
    the standard library, so no vendor package is required.  It is the only
    provider that can analyse images.

``build_provider(config)`` selects one automatically: a real model when an API
key is configured, the deterministic fallback otherwise.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from . import heuristics as h
from . import prompts
from .config import Config

LOGGER = logging.getLogger(__name__)

PERSIAN_LETTER_RE = h.PERSIAN_LETTER_RE


class LLMError(RuntimeError):
    """Raised when a language model call fails irrecoverably."""


class ImageNotSupportedError(LLMError):
    """Raised when image analysis is requested but no vision model exists."""


# --------------------------------------------------------------------------- #
# Response parsing helpers
# --------------------------------------------------------------------------- #

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?")


def parse_json_object(text: str) -> dict[str, Any]:
    """Tolerantly extract the first JSON object from a model response."""
    if not text:
        return {}
    candidates = [text.strip()]
    match = _JSON_BLOCK_RE.search(text)
    if match:
        candidates.insert(0, match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    LOGGER.debug("could not parse JSON from model output: %.200s", text)
    return {}


_SECTION_ALIASES: Mapping[str, tuple[str, ...]] = {
    "definition": (
        "definition", "تعریف", "تعریف ساده", "تعریف:", "توضیح ساده",
    ),
    "intuition": (
        "intuition", "توضیح شهودی", "شهودی", "شعور", "شرح", "توضیح",
        "توضیح ساده", "درک شهودی",
    ),
    "example": ("example", "مثال", "مثال‌ها", "مثالها", "نمونه", "examples"),
    "notes": (
        "key points", "notes", "important points", "نکات مهم", "نکات کلیدی",
        "نکات", "نکته", "نکات اصلی", "کاربردها", "کاربرد",
    ),
    "analysis": ("analysis", "تحلیل مسئله", "تحلیل", "problem analysis", "بررسی مسئله"),
    "method": ("method", "روش حل", "روش", "روش محاسبه"),
    "steps": ("steps", "مراحل حل", "مراحل", "solution steps", "گام‌های حل"),
    "answer": ("final answer", "answer", "پاسخ نهایی", "جواب نهایی", "پاسخ", "نتیجه نهایی"),
    "review": ("check", "review", "بررسی پاسخ", "بررسی", "راستی‌آزمایی"),
}


def parse_labelled_sections(text: str) -> dict[str, str]:
    """Split a ``Label: content`` style model response into a dictionary.

    Headings are matched fuzzily: a leading ``**``/``##``/number/bullet is
    stripped and Persian labels are matched by their core keyword, so
    ``**نکات کلیدی:**`` and ``۳) توضیح شهودی:`` both land in the right bucket.
    When the model uses a label that is not recognised the section is kept
    under the raw label so no content is ever silently dropped.
    """
    sections: dict[str, str] = {}
    current: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current and buffer:
            sections[current] = "\n".join(buffer).strip()

    for raw_line in (text or "").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip().lstrip("#*0123456789۰۱۲۳۴۵۶۷۸۹.)،:- ").strip()
        matched_key: str | None = None
        if ":" in stripped:
            label = stripped.split(":", 1)[0].strip().lower().strip("*# ")
            for key, aliases in _SECTION_ALIASES.items():
                if label in aliases or any(alias in label for alias in aliases):
                    matched_key = key
                    break
            if matched_key is None and len(label) <= 40:
                # keep unknown labels instead of merging them into the previous
                # section (which is how model prose ended up mixed with the
                # offline fallback text)
                matched_key = f"other:{label}"
        if matched_key:
            flush()
            current = matched_key
            buffer = [stripped.split(":", 1)[1].strip()]
        elif current is not None:
            buffer.append(line)
    flush()
    return {k: v for k, v in sections.items() if v}


def split_numbered_steps(text: str) -> list[str]:
    """Extract numbered list items from a block of text."""
    steps: list[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if re.match(r"^[\(\[]?\d+[\)\].\-]\s+", stripped) or re.match(r"^[-*•]\s+", stripped):
            steps.append(re.sub(r"^[\(\[]?\d+[\)\].\-]\s+|^[-*•]\s+", "", stripped).strip())
        elif steps and stripped:
            steps[-1] = f"{steps[-1]} {stripped}".strip()
    return [s for s in steps if s]


def coerce_answer(value: Any) -> Any:
    """Convert a textual answer into a number when that is unambiguous."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if not isinstance(value, str):
        return value
    cleaned = h.normalize_text(value).strip().strip("$").strip()
    cleaned = cleaned.replace(",", "")
    frac = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)", cleaned)
    if frac:
        denominator = float(frac.group(2))
        if denominator:
            return round(float(frac.group(1)) / denominator, 10)
    number = re.fullmatch(r"-?\d+(?:\.\d+)?", cleaned)
    if number:
        return float(cleaned)
    return cleaned


# --------------------------------------------------------------------------- #
# Provider protocol
# --------------------------------------------------------------------------- #

@runtime_checkable
class LLMProvider(Protocol):
    """Interface every language-model backend must implement."""

    name: str
    supports_vision: bool

    def classify(self, query: str, history: Sequence[Mapping[str, str]] | None = None) -> dict[str, Any]:
        """Return routing metadata for *query*."""

    def clarify(self, query: str, missing: str, history: Sequence[Mapping[str, str]] | None = None) -> str:
        """Return a single follow-up question."""

    def explain(
        self,
        query: str,
        topic: str,
        level: str,
        language: str,
        history: Sequence[Mapping[str, str]] | None = None,
    ) -> dict[str, str]:
        """Return concept sections (definition/intuition/example/notes)."""

    def solve(
        self,
        query: str,
        problem_statement: str,
        level: str,
        language: str,
        previous_attempt: Mapping[str, Any] | None = None,
        feedback: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Return an analysed step-by-step solution."""

    def review(
        self,
        problem_statement: str,
        steps: Sequence[str],
        candidate_answer: Any,
        kind: str | None = None,
    ) -> dict[str, Any]:
        """Independently review a proposed answer."""

    def inspect_image(self, image_path: str | Path, question: str | None = None) -> str:
        """Return a transcription/analysis of an image problem."""


# --------------------------------------------------------------------------- #
# Deterministic (offline) provider
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class DeterministicProvider:
    """Offline provider: real mathematics, no network, no API key."""

    name: str = "deterministic"
    supports_vision: bool = False

    # -- routing -------------------------------------------------------- #
    def classify(self, query: str, history: Sequence[Mapping[str, str]] | None = None) -> dict[str, Any]:
        """Classify the query with keyword + expression analysis."""
        normalized = h.normalize_text(query)
        language = h.detect_language(normalized)
        topic = h.detect_topic(normalized)
        level = h.detect_user_level(normalized)
        lowered = normalized.lower()

        expression = h.extract_math_expression(normalized)
        assignments = h.find_assignments(normalized)
        has_equation = bool(expression and "=" in expression)

        solve_words = (
            "حل کن", "حل کنید", "محاسبه", "بدست آور", "به دست آور", "چقدر", "چند است",
            "مقدار", "جواب", "پیدا کن", "چیست اگر",
            "solve", "calculate", "compute", "evaluate", "find", "how much", "what is the value",
        )
        concept_words = (
            "توضیح بده", "توضیح دهید", "چیست", "یعنی چه", "معنی", "تعریف", "چرا", "تفاوت",
            "explain", "what is", "why", "define", "meaning", "difference between", "how does",
        )
        wants_solve = any(word in lowered for word in solve_words)
        wants_concept = any(word in lowered for word in concept_words)

        # "چیست" is ambiguous: "مشتق چیست" is conceptual, "مقدار x چیست" is a solve
        if wants_solve and wants_concept:
            request_type = "mixed" if (assignments or has_equation or wants_solve) else "conceptual"
            if not (assignments or has_equation) and topic != "arithmetic":
                request_type = "mixed" if "solve" in lowered or "حل" in lowered else "conceptual"
        elif wants_solve and (assignments or has_equation or expression):
            request_type = "solvable_problem"
        elif wants_concept:
            request_type = "conceptual"
        elif assignments or has_equation:
            request_type = "solvable_problem"
        elif expression and topic in {"derivative", "integral", "arithmetic", "algebra", "equation"}:
            request_type = "solvable_problem" if topic != "derivative" else "mixed"
        elif expression and not PERSIAN_LETTER_RE.search(normalized):
            # a bare expression such as "12 / 4 + 3" is a problem to evaluate
            request_type = "solvable_problem"
        else:
            request_type = "unclear"

        problem_statement = expression or ""
        if topic == "derivative" and expression:
            request_type = "solvable_problem" if (wants_solve and not wants_concept) else "mixed"
            problem_statement = expression

        missing = h.detect_missing_information(normalized, {
            "request_type": request_type,
            "subject_or_topic": topic,
        })
        is_clear = missing is None
        clarification = "" if is_clear else h.clarification_question(str(missing), language)

        return {
            "request_type": request_type,
            "subject_or_topic": topic,
            "user_level": level,
            "problem_statement": problem_statement,
            "language": language,
            "is_clear": is_clear,
            "missing_information": missing or "none",
            "clarification_question": clarification,
            "confidence": 0.6,
            "provider": self.name,
        }

    def clarify(self, query: str, missing: str, history: Sequence[Mapping[str, str]] | None = None) -> str:
        """Return the canned follow-up question for the missing detail."""
        return h.clarification_question(missing, h.detect_language(query))

    # -- explanation ---------------------------------------------------- #
    def explain(
        self,
        query: str,
        topic: str,
        level: str,
        language: str,
        history: Sequence[Mapping[str, str]] | None = None,
    ) -> dict[str, str]:
        """Return a level-aware concept explanation."""
        return h.explain_concept(topic, query, level, language)

    # -- solving -------------------------------------------------------- #
    def solve(
        self,
        query: str,
        problem_statement: str,
        level: str,
        language: str,
        previous_attempt: Mapping[str, Any] | None = None,
        feedback: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Solve with the deterministic maths engine.

        Variable values are searched for in the problem statement first and in
        the original user message second, so "اگر x=2 باشد، 3x^2 + 1 چقدر است؟"
        is answered 13 even though the router only forwarded the expression.
        """
        text = problem_statement or query
        topic = h.detect_topic(query)
        if topic == "unknown":
            topic = h.detect_topic(text)
        assignments = h.find_assignments(text) or h.find_assignments(query)
        result = h.solve_problem(text, topic, level, language, assignments)
        result["topic"] = topic
        result["provider"] = self.name
        return result

    def review(
        self,
        problem_statement: str,
        steps: Sequence[str],
        candidate_answer: Any,
        kind: str | None = None,
    ) -> dict[str, Any]:
        """Deterministically verify a candidate answer (no model involved)."""
        expression = h.extract_math_expression(problem_statement)
        assignments = h.find_assignments(problem_statement)
        verdict = h.verify_answer(steps, candidate_answer, expression, assignments, kind)
        return {
            "is_correct": verdict["is_correct"],
            "notes": verdict["notes"],
            "corrected_answer": verdict.get("checked"),
            "method": verdict["method"],
        }

    def inspect_image(self, image_path: str | Path, question: str | None = None) -> str:
        """The offline provider cannot see images."""
        raise ImageNotSupportedError(
            "No vision model is configured, so the image cannot be analysed. "
            "Set OPENAI_API_KEY (and optionally MATH_TUTOR_VISION_MODEL) to enable "
            "image support, or type the problem as text."
        )


# --------------------------------------------------------------------------- #
# OpenAI-compatible provider
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class OpenAICompatibleProvider:
    """Minimal stdlib-only client for any OpenAI-compatible chat endpoint."""

    model: str
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.2
    timeout: float = 60.0
    max_tokens: int = 1200
    vision_model: str | None = None
    allow_model_review: bool = True
    name: str = "openai_compatible"
    supports_vision: bool = True

    # -- low level ------------------------------------------------------ #
    def _endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    def _post(self, payload: dict[str, Any], attempts: int = 2) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self._endpoint(),
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:  # pragma: no cover - network
                body = exc.read().decode("utf-8", errors="replace")[:500]
                last_error = LLMError(f"HTTP {exc.code} from model endpoint: {body}")
                if exc.code < 500:
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:  # pragma: no cover
                last_error = LLMError(f"model request failed: {exc}")
            LOGGER.warning("model call attempt %s/%s failed: %s", attempt, attempts, last_error)
        raise last_error or LLMError("model request failed")

    def _chat(self, system: str, user: Any, model: str | None = None) -> str:
        payload = {
            "model": model or self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        response = self._post(payload)
        try:
            return str(response["choices"][0]["message"]["content"] or "")
        except (KeyError, IndexError, TypeError) as exc:  # pragma: no cover - network
            raise LLMError(f"unexpected model response shape: {response}") from exc

    # -- graph facing API ----------------------------------------------- #
    @staticmethod
    def _normalise_topic(raw_topic: str, fallback: str) -> str:
        """Map a free-text topic from the model onto a canonical topic key.

        Models answer with ``"integral"``, ``"انتگرال"``, ``"calculus - integrals"``
        and so on.  Running the same keyword detector used offline turns any of
        those into the canonical key the rest of the graph expects, which is
        what makes the CLI header show a real topic instead of "unknown".
        """
        candidate = (raw_topic or "").strip()
        if not candidate:
            return fallback or "unknown"
        canonical = h.detect_topic(candidate)
        if canonical != "unknown":
            return canonical
        # the model may have answered in English; try the built-in english labels
        for key in h.TOPIC_LABELS_FA:
            if key != "unknown" and key in candidate.lower():
                return key
        return candidate if candidate != "unknown" else (fallback or "unknown")

    def classify(self, query: str, history: Sequence[Mapping[str, str]] | None = None) -> dict[str, Any]:
        """Ask the model for routing metadata, then sanitise it."""
        text = self._chat(prompts.CLASSIFICATION_SYSTEM_PROMPT, prompts.classification_prompt(query, history))
        data = parse_json_object(text)
        fallback = DeterministicProvider().classify(query, history)
        if not data:
            LOGGER.warning("model classification unparsable; falling back to deterministic routing")
            return fallback

        request_type = str(data.get("request_type") or fallback["request_type"]).strip().lower()
        if request_type not in {"conceptual", "solvable_problem", "mixed", "unclear"}:
            request_type = fallback["request_type"]
        level = str(data.get("user_level") or fallback["user_level"]).strip().lower()
        if level not in {"beginner", "intermediate", "advanced", "unknown"}:
            level = fallback["user_level"]
        topic = self._normalise_topic(str(data.get("subject_or_topic") or ""), fallback["subject_or_topic"])
        is_clear = bool(data.get("is_clear", fallback["is_clear"]))
        missing = str(data.get("missing_information") or ("none" if is_clear else "no_math")).strip()
        clarification = str(data.get("clarification_question") or "").strip()
        language = h.detect_language(query)
        if not is_clear:
            # the follow-up question must be in the student's language; the
            # model sometimes slips into English for a Persian message
            wrong_language = language == "fa" and clarification and not h.PERSIAN_LETTER_RE.search(clarification)
            if not clarification or wrong_language:
                clarification = fallback["clarification_question"]

        return {
            "request_type": request_type,
            "subject_or_topic": topic,
            "user_level": level,
            "problem_statement": str(data.get("problem_statement") or fallback["problem_statement"]).strip(),
            "language": language,
            "is_clear": is_clear,
            "missing_information": missing,
            "clarification_question": clarification,
            "confidence": 0.8,
            "provider": self.name,
        }

    def clarify(self, query: str, missing: str, history: Sequence[Mapping[str, str]] | None = None) -> str:
        """Ask the model for one short follow-up question.

        The question must be in the student's language; if the model answers in
        the wrong one the deterministic Persian/English template is used instead
        of showing the student a language they did not write in.
        """
        language = h.detect_language(query)
        text = self._chat(
            prompts.CLARIFY_SYSTEM_PROMPT,
            prompts.clarification_prompt(query, missing, history, language),
        )
        cleaned = (text or "").strip().strip('"')
        if not cleaned:
            return DeterministicProvider().clarify(query, missing, history)

        first_line = next((line.strip() for line in cleaned.splitlines() if line.strip()), "")
        if language == "fa" and not h.PERSIAN_LETTER_RE.search(first_line):
            LOGGER.warning("model answered the clarification in the wrong language; using the template")
            return DeterministicProvider().clarify(query, missing, history)
        return first_line or DeterministicProvider().clarify(query, missing, history)

    def explain(
        self,
        query: str,
        topic: str,
        level: str,
        language: str,
        history: Sequence[Mapping[str, str]] | None = None,
    ) -> dict[str, str]:
        """Ask the model for a concept explanation and normalise the sections.

        The model's own text wins whenever a section is present; the offline
        explanation is used only to fill a section the model genuinely omitted.
        """
        text = self._chat(
            prompts.CONCEPT_SYSTEM_PROMPT,
            prompts.concept_prompt(query, topic, level, language, history),
        )
        sections = parse_labelled_sections(text)
        fallback = h.explain_concept(topic, query, level, language)
        if not sections:
            LOGGER.warning("model explanation unparsable; using deterministic explanation")
            return fallback

        merged: dict[str, str] = {}
        for key in ("definition", "intuition", "example", "notes"):
            value = str(sections.get(key, "")).strip()
            merged[key] = value or fallback[key]

        # surface any extra labelled section the model produced so nothing the
        # model said is lost
        extras = [value for key, value in sections.items() if key.startswith("other:") and value.strip()]
        if extras:
            merged["notes"] = "\n".join([merged["notes"], *extras]).strip()
        return merged

    def solve(
        self,
        query: str,
        problem_statement: str,
        level: str,
        language: str,
        previous_attempt: Mapping[str, Any] | None = None,
        feedback: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Ask the model for a step-by-step solution and normalise the result."""
        text = self._chat(
            prompts.SOLVE_SYSTEM_PROMPT,
            prompts.solve_prompt(query, problem_statement, level, language, previous_attempt, feedback),
        )
        sections = parse_labelled_sections(text)
        steps = split_numbered_steps(sections.get("steps", "")) or [
            line.strip() for line in sections.get("steps", "").splitlines() if line.strip()
        ]
        answer = coerce_answer(sections.get("answer", ""))
        deterministic = DeterministicProvider().solve(query, problem_statement, level, language)
        if not steps:
            LOGGER.warning("model solution unparsable; using deterministic solver")
            return deterministic
        if answer == "" or answer is None:
            answer = deterministic.get("answer")
        return {
            "problem_statement": problem_statement or deterministic.get("problem_statement", ""),
            "given": h.find_assignments(problem_statement or query),
            "unknown": deterministic.get("unknown", ""),
            "method": sections.get("method") or deterministic.get("method", ""),
            "analysis": sections.get("analysis", ""),
            "steps": steps,
            "answer": answer,
            "expression": deterministic.get("expression") or h.extract_math_expression(problem_statement),
            "assignments": deterministic.get("assignments") or {},
            "kind": deterministic.get("kind"),
            "raw": text,
            "provider": self.name,
        }

    def review(
        self,
        problem_statement: str,
        steps: Sequence[str],
        candidate_answer: Any,
        kind: str | None = None,
    ) -> dict[str, Any]:
        """Independent model review, combined with the deterministic check.

        The deterministic recomputation always runs; the model review can only
        *add* a failure, never override a failed recomputation.  This keeps the
        verifier trustworthy even if the model is wrong.
        """
        deterministic = DeterministicProvider().review(problem_statement, steps, candidate_answer, kind)
        if not self.allow_model_review:
            return deterministic
        try:
            text = self._chat(prompts.REVIEW_SYSTEM_PROMPT, prompts.review_prompt(problem_statement, steps, candidate_answer))
        except LLMError as exc:  # pragma: no cover - network
            LOGGER.warning("model review failed (%s); using deterministic verification", exc)
            return deterministic

        data = parse_json_object(text)
        if not data:
            return deterministic
        model_ok = bool(data.get("is_correct", True))
        notes = list(deterministic.get("notes") or [])
        notes.extend(str(note) for note in (data.get("notes") or []) if str(note).strip())
        corrected = data.get("corrected_answer")
        return {
            "is_correct": bool(deterministic["is_correct"]) and model_ok,
            "notes": notes,
            "corrected_answer": coerce_answer(corrected) if corrected is not None else deterministic.get("corrected_answer"),
            "method": f"{deterministic.get('method')}+model_review",
        }

    def inspect_image(self, image_path: str | Path, question: str | None = None) -> str:
        """Send the image to the vision model as a data URL."""
        path = Path(image_path)
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        user_content = [
            {"type": "text", "text": prompts.image_prompt(question)},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
        ]
        return self._chat(prompts.IMAGE_SYSTEM_PROMPT, user_content, model=self.vision_model or self.model)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

def build_provider(config: Config, force_deterministic: bool = False) -> LLMProvider:
    """Return the provider implied by *config*.

    A real model is only used when an API key is present and the caller did not
    explicitly ask for the offline provider.
    """
    resolved = config.resolved_provider
    if force_deterministic or resolved == "deterministic" or not config.api_key:
        if not force_deterministic and resolved != "deterministic":
            LOGGER.warning(
                "no API key configured (provider=%s); using the deterministic offline provider",
                resolved,
            )
        return DeterministicProvider()

    if resolved in {"openai", "openai_compatible"}:
        return OpenAICompatibleProvider(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url or "https://api.openai.com/v1",
            temperature=config.temperature,
            timeout=config.timeout,
            max_tokens=config.max_tokens,
            vision_model=config.vision_model,
            allow_model_review=config.allow_model_review,
            name=resolved,
        )

    if resolved == "ollama":
        return OpenAICompatibleProvider(
            model=config.model,
            api_key=config.api_key or "ollama",
            base_url=config.base_url or "http://localhost:11434/v1",
            temperature=config.temperature,
            timeout=config.timeout,
            max_tokens=config.max_tokens,
            vision_model=config.vision_model,
            allow_model_review=config.allow_model_review,
            name="ollama",
        )

    LOGGER.warning("unknown provider %r; using the deterministic offline provider", resolved)
    return DeterministicProvider()
