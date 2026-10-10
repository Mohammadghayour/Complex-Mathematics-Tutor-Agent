"""Core agent for the Complex Mathematics Tutor.

This module owns **all** of the agent logic: the shared :class:`TutorState`,
the :class:`AgentDeps` dependency bundle, every graph node, the routing
functions that turn the nodes into a graph, and :class:`TutorAgent` which
executes the graph.

Graph shape (mirrors the reference diagram)::

    Start
      |
    read_problem --(image_path?)--> inspect_image --+
      |                                             |
      +---------------------------------------------+
      v
    classify_request
      v
    check_problem_clarity  --(unclear)--> ask_question --(new input)--> read_problem
      |  (clear)
      +--> explain_concept --(mixed)--> solve_problem --> check_answer
      |                                     ^                  |
      |                                     |   (wrong answer) |
      |                                     +--- repair_solution
      |                                                        |
      +-------------------------> format_final_answer <----------+
                                      v
                                 final_answer
                                      v
                                    End

Two execution backends are supported:

* **LangGraph** (preferred, used automatically when installed) - the nodes and
  conditional edges below are registered in a ``langgraph.graph.StateGraph``.
* **Built-in runner** - a small deterministic walker over the very same nodes
  and routing function, used when LangGraph is not installed.  This keeps the
  project runnable and testable in a bare environment.

Both backends execute the identical node functions and the identical router,
so behaviour is the same either way.
"""

from __future__ import annotations

import logging
import logging.handlers
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypedDict

from . import heuristics as h
from .config import (
    IMAGE_MAGIC_NUMBERS,
    MAX_IMAGE_BYTES,
    PROJECT_ROOT,
    SUPPORTED_IMAGE_SUFFIXES,
    Config,
)
from .llm_provider import (
    ImageNotSupportedError,
    LLMError,
    LLMProvider,
    build_provider,
)

try:  # pragma: no cover - exercised only when langgraph is installed
    from langgraph.graph import END, START, StateGraph  # type: ignore

    LANGGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover - default on a bare machine
    END = "__end__"
    START = "__start__"
    StateGraph = None  # type: ignore[assignment]
    LANGGRAPH_AVAILABLE = False

LOGGER = logging.getLogger("complex_math_tutor.agent")

END_NODE = END
START_NODE = START

#: Hard ceiling on node visits per invocation (protects against graph cycles).
MAX_TOTAL_STEPS = 40


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #

class TutorState(TypedDict, total=False):
    """The single structured state threaded through every node.

    ``total=False`` keeps the state easy to build incrementally while the
    documented keys below are always populated by :meth:`TutorAgent.create_initial_state`.
    """

    # --- input -----------------------------------------------------------
    original_query: str
    normalized_query: str
    pending_user_input: str | None
    image_path: str | None
    image_analysis: str | None
    image_processed: bool

    # --- classification --------------------------------------------------
    request_type: str            # conceptual | solvable_problem | mixed | unclear
    problem_statement: str
    subject_or_topic: str
    user_level: str              # beginner | intermediate | advanced | unknown
    language: str                # fa | en
    is_problem_clear: bool
    missing_information: str
    clarification_question: str
    route_decision: str
    classification_meta: dict[str, Any]

    # --- work in progress ------------------------------------------------
    collected_context: list[str]
    solution_steps: list[str]
    candidate_answer: Any
    checked_answer: Any
    is_answer_correct: bool
    verification_notes: list[str]
    verification_method: str
    solution_meta: dict[str, Any]
    concept_sections: dict[str, str]

    # --- output ----------------------------------------------------------
    explanation: str
    final_answer: str
    error_message: str | None

    # --- bookkeeping -----------------------------------------------------
    iteration_count: int
    solve_attempts: int
    awaiting_input: bool
    path_log: list[str]
    last_node: str
    conversation_history: list[dict[str, str]]


@dataclass(slots=True)
class AgentDeps:
    """Everything a node needs besides the state."""

    config: Config
    provider: LLMProvider
    logger: logging.Logger = field(default=LOGGER)
    input_callback: Callable[[str], str] | None = None


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

def setup_logging(debug: bool = False, log_dir: Path | None = None) -> logging.Logger:
    """Configure project logging once and return the package logger.

    A rotating file handler always captures ``DEBUG`` detail; the console
    handler stays at ``WARNING`` (or ``DEBUG`` when *debug* is set) so the CLI
    output remains readable.
    """
    root = logging.getLogger("complex_math_tutor")
    if getattr(root, "_configured", False):
        if debug:
            for handler in root.handlers:
                if isinstance(handler, logging.StreamHandler) and not isinstance(
                    handler, logging.FileHandler
                ):
                    handler.setLevel(logging.DEBUG)
        return root

    root.setLevel(logging.DEBUG)
    root.propagate = False

    target_dir = Path(log_dir) if log_dir else PROJECT_ROOT / "logs"
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            target_dir / "agent.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        root.addHandler(file_handler)
    except OSError:  # pragma: no cover - read-only filesystem
        LOGGER.debug("could not create the log file; continuing with console logging only")

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if debug else logging.WARNING)
    console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    root.addHandler(console)
    root._configured = True  # type: ignore[attr-defined]
    return root


# --------------------------------------------------------------------------- #
# Image handling
# --------------------------------------------------------------------------- #

def resolve_image_path(raw_path: str, config: Config | None = None) -> Path:
    """Resolve a user supplied image path (relative or absolute).

    Candidates are tried in order: the path as typed, the project root and the
    configured ``images`` directory.  The first existing candidate wins; when
    nothing exists the as-typed path is returned so the caller can report it.
    """
    expanded = Path(raw_path).expanduser()
    candidates = [expanded]
    if not expanded.is_absolute():
        candidates.append(PROJECT_ROOT / expanded)
        if config is not None:
            candidates.append(Path(config.images_dir) / expanded)
            candidates.append(Path(config.images_dir) / expanded.name)
        candidates.append(PROJECT_ROOT / "images" / expanded.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return expanded


def validate_image_file(path: Path) -> str | None:
    """Validate an image file. Returns an error message, or ``None`` if valid."""
    if not path.exists():
        return f"تصویر پیدا نشد / image not found: {path}"
    if not path.is_file():
        return f"مسیر داده‌شده یک فایل نیست / not a file: {path}"
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_SUFFIXES:
        return (
            f"فرمت تصویر پشتیبانی نمی‌شود / unsupported image format: {suffix or '(none)'} "
            f"- supported: {', '.join(SUPPORTED_IMAGE_SUFFIXES)}"
        )
    size = path.stat().st_size
    if size == 0:
        return f"فایل تصویر خالی است / empty image file: {path}"
    if size > MAX_IMAGE_BYTES:
        return f"حجم تصویر بیش از حد مجاز است / image too large: {size} bytes"
    expected = IMAGE_MAGIC_NUMBERS.get(suffix, ())
    if expected:
        with path.open("rb") as handle:
            header = handle.read(16)
        if not any(header.startswith(magic) for magic in expected):
            return f"محتوای فایل با پسوند آن هم‌خوانی ندارد / file content does not match extension: {path}"
    return None


# --------------------------------------------------------------------------- #
# Rendering helpers (output shaping - no graph logic)
# --------------------------------------------------------------------------- #

METHOD_LABELS_FA: Mapping[str, str] = {
    "substitution": "جایگذاری مستقیم",
    "evaluation": "محاسبه مستقیم عبارت",
    "expression": "محاسبه مقدار عبارت",
    "equation": "حل معادله",
    "solve equation": "حل معادله",
    "direct evaluation": "محاسبه مستقیم عبارت",
    "evaluate expression": "محاسبه مقدار عبارت",
    "derivative": "مشتق‌گیری با قاعده توان",
    "derivative rule": "مشتق‌گیری با قاعده توان",
}


def method_label(method: str, lang: str = "fa") -> str:
    """Human readable label for an analysis method."""
    if not method:
        return "تحلیل گام‌به‌گام" if lang == "fa" else "step-by-step analysis"
    if lang != "fa":
        return method.replace("_", " ")
    return METHOD_LABELS_FA.get(method, "تحلیل گام‌به‌گام")


def clarify_retry_question(lang: str) -> str:
    """Returned when the clarification budget is exhausted."""
    if lang == "fa":
        return (
            "اطلاعات کافی برای حل دقیق مسئله دریافت نشد. لطفاً صورت مسئله را کامل و "
            "با مقادیر مشخص بنویسید (مثلاً: «اگر x = 2 باشد، مقدار 3x^2 + 1 چقدر است؟»)."
        )
    return (
        "I could not collect enough information to solve this precisely. Please write the "
        "full problem with its values, e.g. 'if x = 2, what is the value of 3x^2 + 1?'."
    )


def render_concept(state: Mapping[str, Any]) -> str:
    """Render the concept sections of the state as terminal-friendly text."""
    sections = state.get("concept_sections") or {}
    lang = state.get("language", "fa")
    fa = lang == "fa"
    headings = {
        "definition": ("تعریف ساده", "Definition"),
        "intuition": ("توضیح شهودی", "Intuition"),
        "example": ("مثال", "Example"),
        "notes": ("نکات مهم", "Key points"),
    }
    blocks: list[str] = []
    for key, (fa_title, en_title) in headings.items():
        body = str(sections.get(key, "")).strip()
        if body:
            blocks.append(f"{fa_title if fa else en_title}:\n{body}")
    return "\n\n".join(blocks) or str(state.get("explanation", "")).strip()


#: Default value of every state field for a brand new turn.
STATE_DEFAULTS: dict[str, Any] = {
    "original_query": "",
    "normalized_query": "",
    "pending_user_input": "",
    "image_path": None,
    "image_analysis": None,
    "image_processed": False,
    "request_type": "unclear",
    "problem_statement": "",
    "subject_or_topic": "unknown",
    "user_level": "unknown",
    "language": "fa",
    "is_problem_clear": False,
    "missing_information": "none",
    "clarification_question": "",
    "route_decision": "",
    "classification_meta": {},
    "collected_context": [],
    "solution_steps": [],
    "candidate_answer": None,
    "checked_answer": None,
    "is_answer_correct": False,
    "verification_notes": [],
    "verification_method": "",
    "solution_meta": {},
    "concept_sections": {},
    "explanation": "",
    "final_answer": "",
    "error_message": None,
    "iteration_count": 0,
    "solve_attempts": 0,
    "awaiting_input": False,
    "path_log": [],
    "last_node": "",
    "conversation_history": [],
}

#: Fields that belong to a single turn and must be cleared when a conversation
#: is resumed with a new question.  ``conversation_history``, ``iteration_count``,
#: ``collected_context``, ``original_query``, ``image_path``, ``subject_or_topic``
#: and ``user_level`` are *not* reset: they carry context forward on purpose.
PER_TURN_FIELDS: tuple[str, ...] = (
    "normalized_query",
    "pending_user_input",
    "image_analysis",
    "image_processed",
    "request_type",
    "problem_statement",
    "is_problem_clear",
    "missing_information",
    "clarification_question",
    "route_decision",
    "classification_meta",
    "solution_steps",
    "candidate_answer",
    "checked_answer",
    "is_answer_correct",
    "verification_notes",
    "verification_method",
    "solution_meta",
    "concept_sections",
    "explanation",
    "final_answer",
    "error_message",
    "solve_attempts",
    "awaiting_input",
    "path_log",
    "last_node",
)


def reset_turn(state: TutorState) -> TutorState:
    """Clear the per-turn fields of *state*, keeping the conversation context.

    Used by :meth:`TutorAgent.run` when an existing state is passed back in for
    a new question, so answers from the previous turn cannot leak into the next
    one (for example a stale ``concept_sections`` block).
    """
    fresh: dict[str, Any] = dict(state)
    for field_name in PER_TURN_FIELDS:
        default = STATE_DEFAULTS[field_name]
        fresh[field_name] = list(default) if isinstance(default, list) else (
            dict(default) if isinstance(default, dict) else default
        )
    return fresh  # type: ignore[return-value]


def _format_answer(value: Any) -> str:
    """Render the checked answer, degrading gracefully when it is missing."""
    if value is None or value == "":
        return "(پاسخ عددی محاسبه نشد / no numeric answer was produced)"
    return h.format_number(value)


def render_solution(state: Mapping[str, Any]) -> str:
    """Render the solution part of the state as terminal-friendly text."""
    lang = state.get("language", "fa")
    fa = lang == "fa"
    steps: Sequence[str] = state.get("solution_steps") or []
    meta: Mapping[str, Any] = state.get("solution_meta") or {}
    given = meta.get("given") or {}
    unknown = meta.get("unknown") or ("مقدار نهایی" if fa else "the final value")
    method = meta.get("method") or ""
    method_text = method_label(str(method), lang)
    analysis_text = str(meta.get("analysis") or "").strip()

    lines: list[str] = []
    lines.append("تحلیل مسئله:" if fa else "Problem analysis:")
    lines.append(f"- صورت مسئله / problem: {state.get('problem_statement') or state.get('normalized_query', '')}")
    if given:
        given_text = ", ".join(f"{k} = {h.format_number(v)}" for k, v in given.items())
        lines.append(("- داده‌ها / given: " if fa else "- Given: ") + given_text)
    lines.append(("- خواسته / asked: " if fa else "- Asked: ") + str(unknown))
    lines.append(("- روش حل / method: " if fa else "- Method: ") + str(method_text))
    if analysis_text:
        lines.append(f"- {analysis_text}")

    lines.append("")
    lines.append("مراحل حل:" if fa else "Solution steps:")
    if steps:
        for index, step in enumerate(steps, start=1):
            lines.append(f"{index}. {step}")
    else:
        lines.append(("1. گام عددی مشخصی تولید نشد." if fa else "1. No numeric step was produced."))

    lines.append("")
    lines.append("پاسخ نهایی:" if fa else "Final answer:")
    lines.append(_format_answer(state.get("checked_answer", state.get("candidate_answer"))))

    notes = state.get("verification_notes") or []
    lines.append("")
    lines.append("بررسی پاسخ:" if fa else "Answer check:")
    if notes:
        for note in notes:
            lines.append(f"- {note}")
    else:
        lines.append("- بررسی مستقل انجام نشد / no independent check was performed.")
    verdict = state.get("is_answer_correct")
    lines.append(
        ("- نتیجه بررسی / verification: " + ("درست / correct" if verdict else "نیازمند بازبینی / needs review"))
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #

def read_problem(state: TutorState, deps: AgentDeps) -> dict[str, Any]:
    """Ingest, normalise and register the incoming user message.

    Consumes ``pending_user_input`` (set by the CLI or by ``ask_question``),
    normalises it, appends it to the conversation history and performs the
    first cheap extraction pass (topic / level).
    """
    raw = (state.get("pending_user_input") or "").strip()
    if not raw:
        language = state.get("language", "fa")
        return {
            "error_message": "ورودی خالی است / empty input",
            "is_problem_clear": False,
            "request_type": "unclear",
            "missing_information": "no_math",
            "clarification_question": h.clarification_question("no_math", language),
        }

    normalized = h.normalize_text(raw)
    original = state.get("original_query") or raw
    language = state.get("language") or h.detect_language(normalized)
    # an empty placeholder from a fresh state must not be mistaken for a topic
    topic = h.detect_topic(normalized) or str(state.get("subject_or_topic") or "unknown")
    level = h.detect_user_level(normalized)

    history = list(state.get("conversation_history") or [])
    history.append({"role": "user", "content": raw})

    updates: dict[str, Any] = {
        "original_query": original,
        "normalized_query": normalized,
        "pending_user_input": None,
        "language": language,
        "conversation_history": history,
        "error_message": None,
    }
    if topic != "unknown":
        updates["subject_or_topic"] = topic
    if level != "unknown":
        updates["user_level"] = level
    if state.get("iteration_count", 0) > 0:
        # only clarifications accumulate as extra context
        collected = list(state.get("collected_context") or [])
        collected.append(normalized)
        updates["collected_context"] = collected
    if state.get("image_path") and not state.get("problem_statement"):
        updates["problem_statement"] = normalized
    return updates


def inspect_image(state: TutorState, deps: AgentDeps) -> dict[str, Any]:
    """Validate the supplied image and ask the vision model to read it.

    Never crashes the graph: when the file is invalid or no vision model is
    configured, an understandable ``error_message`` is stored and the flow
    continues (the agent then asks the user to type the problem instead).
    """
    raw_path = str(state.get("image_path") or "")
    resolved = resolve_image_path(raw_path, deps.config)
    error = validate_image_file(resolved)
    if error:
        deps.logger.warning("image validation failed: %s", error)
        return {
            "image_processed": True,
            "image_analysis": None,
            "error_message": error,
            "is_problem_clear": False,
            "missing_information": "no_math",
            "clarification_question": h.clarification_question("no_math", state.get("language", "fa")),
        }

    try:
        analysis = deps.provider.inspect_image(resolved, state.get("normalized_query") or None)
    except ImageNotSupportedError as exc:
        deps.logger.warning("vision unavailable: %s", exc)
        return {
            "image_processed": True,
            "image_analysis": None,
            "image_path": str(resolved),
            "error_message": str(exc),
            "is_problem_clear": False,
            "missing_information": "no_math",
            "clarification_question": h.clarification_question("no_math", state.get("language", "fa")),
        }
    except LLMError as exc:  # pragma: no cover - network
        return {
            "image_processed": True,
            "image_analysis": None,
            "image_path": str(resolved),
            "error_message": f"تحلیل تصویر ناموفق بود / image analysis failed: {exc}",
            "is_problem_clear": False,
            "missing_information": "no_math",
            "clarification_question": h.clarification_question("no_math", state.get("language", "fa")),
        }

    combined = "\n".join(part for part in [state.get("normalized_query") or "", analysis] if part.strip())
    deps.logger.debug("image analysed: %.120s", analysis)
    return {
        "image_processed": True,
        "image_path": str(resolved),
        "image_analysis": analysis,
        "problem_statement": analysis.strip(),
        "normalized_query": combined.strip(),
        "error_message": None,
        "is_problem_clear": True,
    }


def classify_request(state: TutorState, deps: AgentDeps) -> dict[str, Any]:
    """Ask the provider to classify the request and extract routing metadata.

    Fills ``request_type``, ``subject_or_topic``, ``user_level``,
    ``problem_statement``, ``is_problem_clear``, ``missing_information`` and a
    first draft of ``clarification_question``.  Unknown levels default to
    ``beginner`` so the explanation is never pitched above the user.
    """
    query = state.get("normalized_query") or state.get("original_query") or ""
    history = state.get("conversation_history") or []
    try:
        meta = deps.provider.classify(query, history)
    except LLMError as exc:  # pragma: no cover - network
        deps.logger.error("classification failed: %s", exc)
        meta = {}
        state = dict(state)
        state["error_message"] = f"طبقه‌بندی درخواست ناموفق بود / classification failed: {exc}"
        return {"error_message": state["error_message"], "request_type": "unclear", "is_problem_clear": False}

    request_type = str(meta.get("request_type") or "unclear")
    level = str(meta.get("user_level") or "unknown")
    if level == "unknown":
        level = str(state.get("user_level") or "unknown")
    if level == "unknown":
        level = "beginner" if request_type != "unclear" else "unknown"

    topic = str(meta.get("subject_or_topic") or state.get("subject_or_topic") or "unknown")
    problem_statement = str(meta.get("problem_statement") or "").strip()
    if not problem_statement:
        problem_statement = str(state.get("problem_statement") or "")
    if not problem_statement:
        problem_statement = h.extract_math_expression(query) or ""

    return {
        "request_type": request_type,
        "subject_or_topic": topic,
        "user_level": level,
        "problem_statement": problem_statement,
        "language": str(meta.get("language") or state.get("language") or "fa"),
        "is_problem_clear": bool(meta.get("is_clear", False)),
        "missing_information": str(meta.get("missing_information") or "none"),
        "clarification_question": str(meta.get("clarification_question") or ""),
        "classification_meta": dict(meta),
    }


def check_problem_clarity(state: TutorState, deps: AgentDeps) -> dict[str, Any]:
    """Decide whether the request can be answered, and pick the next branch.

    This is the graph's real branch point: it re-validates the classification
    against the actual state, forces ``unclear`` for unusable input and writes
    ``route_decision`` which the conditional edge consumes.
    """
    request_type = str(state.get("request_type") or "unclear")
    problem_statement = str(state.get("problem_statement") or "")
    missing = str(state.get("missing_information") or "none")
    language = state.get("language", "fa")

    is_clear = bool(state.get("is_problem_clear", False))
    if request_type == "unclear":
        is_clear = False
        missing = "none" if missing != "none" else "no_math"
        if missing == "none":
            missing = "no_math"
    query = state.get("normalized_query") or ""
    expression = h.extract_math_expression(problem_statement)
    assignments = h.find_assignments(query)
    if not expression:
        expression = h.extract_math_expression(query)
    if request_type in {"solvable_problem", "mixed"}:
        if not problem_statement:
            is_clear = False
            missing = "no_expression"
        elif assignments and not h.expression_covers_assignment(expression, assignments):
            # the user pinned a value but the expression still needs another
            # unknown -> a genuine clarification round, not a guess
            is_clear = False
            missing = "no_variable_value"
    if state.get("image_analysis"):
        is_clear = True
    if state.get("error_message") and not problem_statement:
        is_clear = False

    clarification = str(state.get("clarification_question") or "")
    if not is_clear and not clarification:
        try:
            clarification = deps.provider.clarify(
                state.get("normalized_query") or "", missing, state.get("conversation_history") or []
            )
        except LLMError as exc:  # pragma: no cover - network
            deps.logger.warning("clarification generation failed: %s", exc)
            clarification = h.clarification_question(missing, language)

    if is_clear:
        route = {
            "conceptual": "explain_concept",
            "mixed": "explain_concept",
            "solvable_problem": "solve_problem",
        }.get(request_type, "explain_concept")
    else:
        route = "ask_question"

    deps.logger.debug(
        "clarity check: type=%s clear=%s missing=%s route=%s", request_type, is_clear, missing, route
    )
    return {
        "is_problem_clear": is_clear,
        "missing_information": missing,
        "clarification_question": clarification,
        "route_decision": route,
    }


def ask_question(state: TutorState, deps: AgentDeps) -> dict[str, Any]:
    """Produce a follow-up question and collect the user's answer.

    When an ``input_callback`` is available (interactive CLI, scripted tests)
    the answer is fetched immediately and the graph loops back to
    ``read_problem``.  Otherwise the turn ends with the question and
    ``awaiting_input`` set, so the CLI can ask the user and resume.
    """
    language = state.get("language", "fa")
    question = str(state.get("clarification_question") or "").strip()
    if not question:
        question = h.clarification_question(str(state.get("missing_information") or "no_math"), language)

    iteration = int(state.get("iteration_count", 0)) + 1
    if iteration > int(deps.config.max_clarification_rounds):
        # budget exhausted: stop asking and tell the user what is missing
        retry = clarify_retry_question(language)
        history = list(state.get("conversation_history") or [])
        history.append({"role": "assistant", "content": retry})
        return {
            "clarification_question": retry,
            "iteration_count": iteration,
            "conversation_history": history,
            "pending_user_input": None,
            "explanation": retry,
            "final_answer": retry,
            "awaiting_input": False,
            "route_decision": "final_answer",
        }

    history = list(state.get("conversation_history") or [])
    history.append({"role": "assistant", "content": question})

    pending: str | None = None
    can_continue = deps.input_callback is not None
    if can_continue:
        try:
            answer = (deps.input_callback or (lambda _: ""))(question)
        except (EOFError, KeyboardInterrupt):  # pragma: no cover - interactive
            answer = ""
        except Exception as exc:  # pragma: no cover - defensive
            deps.logger.warning("input callback failed: %s", exc)
            answer = ""
        if answer and answer.strip():
            pending = answer.strip()

    return {
        "clarification_question": question,
        "iteration_count": iteration,
        "conversation_history": history,
        "pending_user_input": pending,
        "explanation": question,
        "final_answer": question,
        "awaiting_input": pending is None,
        "route_decision": "read_problem" if pending else "final_answer",
    }


def explain_concept(state: TutorState, deps: AgentDeps) -> dict[str, Any]:
    """Explain the concept behind the request, tuned to the user's level."""
    query = state.get("normalized_query") or ""
    topic = str(state.get("subject_or_topic") or "unknown")
    level = str(state.get("user_level") or "beginner")
    language = state.get("language", "fa")
    try:
        sections = deps.provider.explain(query, topic, level, language, state.get("conversation_history") or [])
    except LLMError as exc:  # pragma: no cover - network
        deps.logger.error("concept explanation failed: %s", exc)
        sections = h.explain_concept(topic, query, level, language)

    preview = dict(state)
    preview["concept_sections"] = sections
    return {
        "concept_sections": dict(sections),
        "explanation": render_concept(preview),
        "error_message": None,
    }


def solve_problem(state: TutorState, deps: AgentDeps) -> dict[str, Any]:
    """Split the problem into logical steps and produce a candidate answer."""
    query = state.get("normalized_query") or ""
    previous_statement = str(state.get("problem_statement") or "")
    level = str(state.get("user_level") or "beginner")
    language = state.get("language", "fa")
    attempts = int(state.get("solve_attempts", 0)) + 1

    previous = None
    if state.get("solution_steps"):
        previous = {
            "candidate_answer": state.get("candidate_answer"),
            "solution_steps": state.get("solution_steps"),
        }

    try:
        solution = deps.provider.solve(query, previous_statement, level, language, previous_attempt=previous)
    except LLMError as exc:  # pragma: no cover - network
        deps.logger.error("solving failed: %s", exc)
        return {
            "error_message": f"حل مسئله ناموفق بود / solving failed: {exc}",
            "solve_attempts": attempts,
        }

    steps = [str(step).strip() for step in (solution.get("steps") or []) if str(step).strip()]
    # keep the problem statement the provider used (it may be more precise than
    # what the router extracted, e.g. after an image transcription)
    problem = str(solution.get("problem_statement") or "").strip() or previous_statement
    return {
        "problem_statement": problem,
        "solution_steps": steps,
        "candidate_answer": solution.get("answer"),
        "solution_meta": dict(solution),
        "solve_attempts": attempts,
        "error_message": None,
    }


def check_answer(state: TutorState, deps: AgentDeps) -> dict[str, Any]:
    """Independently verify the produced answer and record the verdict.

    Verification combines (a) recomputation of the expression with Python,
    (b) consistency of every numeric step and (c) - when a model is available -
    a separate review call.  ``is_answer_correct`` is always written.

    When the provider already verified its own output (the offline engine does:
    it recomputes every result and cross-checks derivatives numerically) that
    verdict is used directly instead of being recomputed a second time.
    """
    steps = list(state.get("solution_steps") or [])
    candidate = state.get("candidate_answer")
    problem = str(state.get("problem_statement") or "")
    meta = state.get("solution_meta") or {}
    kind = str(meta.get("kind") or "") or None

    if candidate is None and not steps:
        verdict = {
            "is_correct": False,
            "notes": ["پاسخ یا مراحل حل تولید نشد / no answer or steps were produced."],
            "corrected_answer": None,
            "method": "structural",
        }
    elif meta.get("is_answer_correct") is not None and getattr(deps.provider, "name", "") == "deterministic":
        verdict = {
            "is_correct": bool(meta.get("is_answer_correct")),
            "notes": list(meta.get("verification_notes") or []),
            "corrected_answer": candidate,
            "method": str(meta.get("verification_method") or "recomputation"),
        }
    else:
        expression = h.extract_math_expression(problem)
        assignments = h.find_assignments(problem)
        try:
            verdict = deps.provider.review(problem, steps, candidate, kind)
        except LLMError as exc:  # pragma: no cover - network
            deps.logger.warning("review failed (%s); falling back to local verification", exc)
            local = h.verify_answer(
                steps,
                candidate,
                h.extract_math_expression(problem),
                h.find_assignments(problem),
                kind,
            )
            verdict = {
                "is_correct": local["is_correct"],
                "notes": local["notes"],
                "corrected_answer": local.get("checked"),
                "method": local["method"],
            }
    is_correct = bool(verdict.get("is_correct"))
    checked = verdict.get("corrected_answer", candidate)
    if not is_correct and checked is None:
        checked = candidate

    return {
        "is_answer_correct": is_correct,
        "checked_answer": checked,
        "verification_notes": [str(note) for note in (verdict.get("notes") or [])],
        "verification_method": str(verdict.get("method") or "structural"),
        # keep an earlier failure (e.g. a solver outage) visible in the state
        "error_message": state.get("error_message"),
    }


def repair_solution(state: TutorState, deps: AgentDeps) -> dict[str, Any]:
    """Re-solve a failed answer, feeding the checker's feedback back to the solver."""
    query = state.get("normalized_query") or ""
    problem = str(state.get("problem_statement") or query)
    level = str(state.get("user_level") or "beginner")
    language = state.get("language", "fa")
    feedback = [str(note) for note in (state.get("verification_notes") or [])]
    attempts = int(state.get("solve_attempts", 0)) + 1

    deps.logger.info("repairing solution (attempt %s): %s", attempts, feedback)
    try:
        solution = deps.provider.solve(
            query,
            problem,
            level,
            language,
            previous_attempt={
                "candidate_answer": state.get("candidate_answer"),
                "solution_steps": state.get("solution_steps"),
            },
            feedback=feedback,
        )
    except LLMError as exc:  # pragma: no cover - network
        deps.logger.error("repair failed: %s", exc)
        return {"solve_attempts": attempts, "error_message": f"اصلاح پاسخ ناموفق بود / repair failed: {exc}"}

    steps = [str(step).strip() for step in (solution.get("steps") or []) if str(step).strip()]
    answer = solution.get("answer")
    if answer is None:
        answer = state.get("checked_answer", state.get("candidate_answer"))
    meta = dict(solution)
    # the repair feedback is diagnostic material, not a solution step
    meta["repair_feedback"] = feedback
    return {
        "solution_steps": steps or list(state.get("solution_steps") or []),
        "candidate_answer": answer,
        "solution_meta": meta,
        "solve_attempts": attempts,
        "error_message": None,
    }


def format_final_answer(state: TutorState, deps: AgentDeps) -> dict[str, Any]:
    """Shape the verified material into the answer the user finally sees.

    Solvable problems get *problem analysis / method / steps / final answer /
    verification*; conceptual questions get *definition / intuition / example /
    key points*; mixed requests get the concept first and the solution after.
    """
    request_type = str(state.get("request_type") or "conceptual")
    language = state.get("language", "fa")
    fa = language == "fa"
    blocks: list[str] = []

    if state.get("error_message"):
        blocks.append(
            ("توجه: " if fa else "Note: ") + str(state["error_message"])
        )

    if state.get("concept_sections"):
        concept = render_concept(state)
        if concept:
            blocks.append(concept)

    has_solution = bool(state.get("solution_steps")) or state.get("candidate_answer") is not None
    if has_solution and request_type in {"solvable_problem", "mixed"}:
        blocks.append(render_solution(state))

    if not blocks:
        blocks.append(
            str(state.get("explanation") or state.get("final_answer") or "")
            or ("پاسخی تولید نشد." if fa else "No answer was produced.")
        )

    text = "\n\n".join(block for block in blocks if block.strip())
    history = list(state.get("conversation_history") or [])
    history.append({"role": "assistant", "content": text})
    return {"final_answer": text, "conversation_history": history, "awaiting_input": False}


def final_answer(state: TutorState, deps: AgentDeps) -> dict[str, Any]:
    """Terminate the graph, guaranteeing that ``final_answer`` is populated.

    When the clarification budget was exhausted the graph arrives here without
    an answer, so a final, actionable message is produced instead of repeating
    the last follow-up question.
    """
    text = str(state.get("final_answer") or "").strip()
    if not text:
        text = str(state.get("explanation") or "").strip()
    if not text:
        language = state.get("language", "fa")
        if state.get("iteration_count", 0) > 0:
            text = clarify_retry_question(language)
        else:
            text = "پاسخی تولید نشد." if language == "fa" else "No answer was produced."
    return {
        "final_answer": text,
        "awaiting_input": bool(state.get("awaiting_input", False)),
    }


#: Node registry - the graph is built from this mapping.
NODES: Mapping[str, Callable[[TutorState, AgentDeps], dict[str, Any]]] = {
    "read_problem": read_problem,
    "inspect_image": inspect_image,
    "classify_request": classify_request,
    "check_problem_clarity": check_problem_clarity,
    "ask_question": ask_question,
    "explain_concept": explain_concept,
    "solve_problem": solve_problem,
    "check_answer": check_answer,
    "repair_solution": repair_solution,
    "format_final_answer": format_final_answer,
    "final_answer": final_answer,
}


# --------------------------------------------------------------------------- #
# Routing (conditional edges)
# --------------------------------------------------------------------------- #

def next_node(current: str, state: TutorState, deps: AgentDeps) -> str:
    """Return the next node name for *current* - the single source of routing truth.

    Used by the built-in runner directly and registered as the conditional-edge
    function for every branching node of the LangGraph backend.
    """
    max_clarifications = int(deps.config.max_clarification_rounds)
    max_solves = int(deps.config.max_solve_attempts)
    iterations = int(state.get("iteration_count", 0))

    if current == "read_problem":
        if state.get("image_path") and not state.get("image_processed"):
            return "inspect_image"
        return "classify_request"

    if current == "inspect_image":
        return "classify_request"

    if current == "classify_request":
        return "check_problem_clarity"

    if current == "check_problem_clarity":
        if state.get("is_problem_clear"):
            return str(state.get("route_decision") or "explain_concept")
        if iterations >= max_clarifications:
            deps.logger.info("clarification budget exhausted (%s rounds)", iterations)
            return "final_answer"
        return "ask_question"

    if current == "ask_question":
        if state.get("pending_user_input") and iterations < max_clarifications:
            return "read_problem"
        return "final_answer"

    if current == "explain_concept":
        if state.get("request_type") == "mixed" and state.get("problem_statement"):
            return "solve_problem"
        return "format_final_answer"

    if current == "solve_problem":
        return "check_answer"

    if current == "check_answer":
        if not state.get("is_answer_correct") and int(state.get("solve_attempts", 0)) < max_solves:
            return "repair_solution"
        return "format_final_answer"

    if current == "repair_solution":
        return "check_answer"

    if current == "format_final_answer":
        return "final_answer"

    if current == "final_answer":
        return END_NODE

    raise ValueError(f"unknown node: {current}")


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #

class TutorAgent:
    """The Complex Mathematics Tutor agent.

    Parameters
    ----------
    config:
        Runtime configuration; read from the environment when omitted.
    provider:
        Explicit LLM provider (mainly for tests).  Built from *config* otherwise.
    input_callback:
        Optional callable used by ``ask_question`` to collect a follow-up answer
        during a single graph run.  The interactive CLI passes one; tests pass a
        scripted one.
    """

    def __init__(
        self,
        config: Config | None = None,
        provider: LLMProvider | None = None,
        input_callback: Callable[[str], str] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config or Config.from_env()
        self.logger = logger or setup_logging(self.config.debug)
        self.provider = provider or build_provider(self.config)
        self.input_callback = input_callback

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def backend(self) -> str:
        """Which execution backend is in use (``langgraph`` or ``builtin``)."""
        return "langgraph" if LANGGRAPH_AVAILABLE else "builtin"

    @property
    def provider_name(self) -> str:
        """Name of the active provider."""
        return getattr(self.provider, "name", "unknown")

    @property
    def deps(self) -> AgentDeps:
        """Dependency bundle used for the current turn (read-only helper)."""
        return AgentDeps(
            config=self.config,
            provider=self.provider,
            logger=self.logger,
            input_callback=self.input_callback,
        )

    def describe(self) -> str:
        """Multi-line description of the agent used by the CLI banner."""
        return (
            f"graph backend : {self.backend}\n"
            f"provider      : {self.provider_name}\n"
            f"model         : {self.config.model if self.config.has_language_model else '(offline)'}\n"
            f"vision        : {'yes' if getattr(self.provider, 'supports_vision', False) else 'no'}\n"
            f"levels        : beginner | intermediate | advanced\n"
            f"max clarify   : {self.config.max_clarification_rounds} rounds\n"
            f"max solve try : {self.config.max_solve_attempts} attempts"
        )

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #
    def create_initial_state(
        self,
        query: str,
        image_path: str | None = None,
        language: str | None = None,
    ) -> TutorState:
        """Build a fully populated :class:`TutorState` for a new turn."""
        state: TutorState = {
            key: (list(value) if isinstance(value, list) else dict(value) if isinstance(value, dict) else value)
            for key, value in STATE_DEFAULTS.items()
        }  # type: ignore[assignment]
        state["original_query"] = query or ""
        state["normalized_query"] = h.normalize_text(query or "")
        state["pending_user_input"] = query or ""
        state["image_path"] = image_path
        state["language"] = language or h.detect_language(query or "")
        return state

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    def run(
        self,
        query: str,
        image_path: str | None = None,
        input_callback: Callable[[str], str] | None = None,
        state: TutorState | None = None,
        language: str | None = None,
    ) -> TutorState:
        """Run the graph for one user turn and return the final state.

        Passing an existing *state* resumes a conversation: the previous
        history, clarification counter and collected context are preserved.
        """
        callback = input_callback if input_callback is not None else self.input_callback
        deps = AgentDeps(
            config=self.config,
            provider=self.provider,
            logger=self.logger,
            input_callback=callback,
        )
        if state:
            # resuming a conversation: keep the history/context, clear the
            # per-turn fields so the previous answer cannot leak into this one
            working = reset_turn(state)
            working["original_query"] = query or working.get("original_query") or ""
        else:
            working = dict(self.create_initial_state(query, image_path, language))
        working["pending_user_input"] = query or working.get("pending_user_input") or ""
        if image_path is not None:
            working["image_path"] = image_path
            working["image_processed"] = False
            working["image_analysis"] = None
        working["awaiting_input"] = False

        self.logger.debug(
            "graph start | backend=%s provider=%s query=%.80s image=%s",
            self.backend,
            self.provider_name,
            query,
            working.get("image_path"),
        )
        result = self._run_langgraph(working, deps) if LANGGRAPH_AVAILABLE else self._run_builtin(working, deps)
        result["path_log"] = list(result.get("path_log") or [])
        self.logger.debug("graph finished | path=%s", " -> ".join(result["path_log"]))
        return result  # type: ignore[return-value]

    # -- built-in runner ------------------------------------------------- #
    def _run_builtin(self, state: TutorState, deps: AgentDeps) -> TutorState:
        """Deterministic walker over :data:`NODES` using :func:`next_node`."""
        working: dict[str, Any] = dict(state)
        working["path_log"] = list(working.get("path_log") or [])
        current = "read_problem"
        for _ in range(MAX_TOTAL_STEPS):
            node = NODES.get(current)
            if node is None:
                raise ValueError(f"unknown node in graph: {current}")
            updates = node(working, deps) or {}  # type: ignore[arg-type]
            working.update(updates)
            working["last_node"] = current
            working["path_log"] = list(working.get("path_log") or []) + [current]
            current = next_node(current, working, deps)  # type: ignore[arg-type]
            if current == END_NODE:
                return working  # type: ignore[return-value]
        raise RuntimeError(
            f"graph did not terminate within {MAX_TOTAL_STEPS} steps; path={working['path_log']}"
        )

    # -- langgraph runner ------------------------------------------------ #
    def _build_langgraph(self, deps: AgentDeps):  # pragma: no cover - requires langgraph
        """Compile the LangGraph state machine (nodes + conditional edges)."""
        graph = StateGraph(TutorState)  # type: ignore[misc]
        for name, node in NODES.items():
            graph.add_node(name, self._wrap_node(name, node, deps))
        graph.add_edge(START, "read_problem")
        for name in NODES:
            if name == "final_answer":
                graph.add_edge(name, END)
                continue
            graph.add_conditional_edges(name, self._route_for(name, deps))
        return graph.compile()

    def _wrap_node(
        self,
        name: str,
        node: Callable[[TutorState, AgentDeps], dict[str, Any]],
        deps: AgentDeps,
    ) -> Callable[[TutorState], dict[str, Any]]:  # pragma: no cover - requires langgraph
        """Adapt a ``(state, deps)`` node into a LangGraph node."""

        def wrapped(state: TutorState) -> dict[str, Any]:
            updates = node(state, deps) or {}
            updates["last_node"] = name
            updates["path_log"] = list(state.get("path_log") or []) + [name]
            return updates

        return wrapped

    def _route_for(self, name: str, deps: AgentDeps) -> Callable[[TutorState], str]:  # pragma: no cover
        """Build the conditional-edge function for *name*."""

        def route(state: TutorState) -> str:
            return next_node(name, state, deps)

        return route

    def _run_langgraph(self, state: TutorState, deps: AgentDeps) -> TutorState:  # pragma: no cover
        """Execute the compiled LangGraph state machine."""
        app = self._build_langgraph(deps)
        return app.invoke(state, config={"recursion_limit": MAX_TOTAL_STEPS * 2})  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    # Convenience helpers
    # ------------------------------------------------------------------ #
    def run_turns(
        self,
        turns: Sequence[str],
        image_path: str | None = None,
    ) -> list[TutorState]:
        """Run several user turns, keeping the conversation context between them."""
        states: list[TutorState] = []
        state: TutorState | None = None
        for index, turn in enumerate(turns):
            state = self.run(turn, image_path=image_path if index == 0 else None, state=state)
            states.append(state)
        return states

    def describe_turn(self, state: Mapping[str, Any]) -> str:
        """Short human-readable summary of a completed turn (CLI header)."""
        lang = state.get("language", "fa")
        request_type = str(state.get("request_type") or "unclear")
        topic = str(state.get("subject_or_topic") or "unknown")
        level = str(state.get("user_level") or "unknown")
        lines = [
            f"نوع درخواست: {h.request_label(request_type, lang)}",
            f"موضوع: {h.topic_label(topic, lang)}",
            f"سطح تقریبی: {h.level_label(level, lang)}",
        ]
        if state.get("awaiting_input"):
            lines.append("وضعیت: منتظر پاسخ تکمیلی کاربر")
        return "\n".join(lines)
