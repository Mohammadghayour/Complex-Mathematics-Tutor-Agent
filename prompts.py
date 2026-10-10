"""Prompt templates for the Complex Mathematics Tutor Agent.

All natural-language instructions sent to a language model live here so that
the graph logic in :mod:`agent` stays free of prose.  Every prompt is built by
a small pure function returning a ``str`` (or a list of chat messages), which
makes them trivially unit-testable.

The agent must behave identically with or without a real model, therefore the
prompts are written to be *strict* about the answer shape:

* classification prompts must answer with a single JSON object;
* solution prompts must return numbered steps plus one explicit final answer
  line so the independent checker in :mod:`heuristics` can verify it.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

CLASSIFICATION_SYSTEM_PROMPT = """\
You are the routing brain of a mathematics tutoring agent.
Classify the student's message. Answer ONLY with a JSON object using these keys:
{
  "request_type": "conceptual" | "solvable_problem" | "mixed" | "unclear",
  "subject_or_topic": "short topic label",
  "user_level": "beginner" | "intermediate" | "advanced" | "unknown",
  "problem_statement": "the maths problem restated precisely, or empty string",
  "is_clear": true | false,
  "missing_information": "one of no_math | no_expression | no_variable_value | unknown_topic | none",
  "clarification_question": "a short follow-up question when is_clear is false, else empty string"
}
Rules:
- "conceptual": the student asks what a concept means or why it works.
- "solvable_problem": the student gives concrete data and wants a numeric answer.
- "mixed": both an explanation and a computation are requested.
- "unclear": not enough mathematics to answer.
- Never invent numbers the student did not provide.
"""


def classification_prompt(query: str, history: Sequence[Mapping[str, str]] | None = None) -> str:
    """Build the user message for the classification node."""
    parts = []
    if history:
        transcript = "\n".join(
            f"{item.get('role', 'user')}: {item.get('content', '')}" for item in list(history)[-6:]
        )
        parts.append(f"Conversation so far:\n{transcript}")
    parts.append(f"Student message:\n{query}")
    parts.append("Return the JSON object only.")
    return "\n\n".join(parts)


CLARIFY_SYSTEM_PROMPT = """\
You are a friendly mathematics tutor. The student's request is missing
information. Write ONE short follow-up question (max 25 words) in the same
language as the student, asking for exactly the missing detail. Ask only one
thing. Do not answer the question yet.
"""


def clarification_prompt(
    query: str,
    missing: str,
    history: Sequence[Mapping[str, str]] | None = None,
    language: str | None = None,
) -> str:
    """Build the user message for the ask_question node."""
    transcript = ""
    if history:
        transcript = "\n".join(
            f"{item.get('role', 'user')}: {item.get('content', '')}" for item in list(history)[-6:]
        )
    lang = language or h.detect_language(query)
    if lang == "fa":
        language_rule = "Write the question in PERSIAN (فارسی). Do not answer in English."
    else:
        language_rule = "Write the question in English."
    return (
        f"Conversation so far:\n{transcript}\n\n"
        f"Student message:\n{query}\n\n"
        f"Missing information code: {missing}\n"
        f"{language_rule}\n"
        "Write the single follow-up question."
    )


CONCEPT_SYSTEM_PROMPT = """\
You are a mathematics tutor explaining a concept.
Match the student's level and language. Be precise but simple.
Answer with these four labelled sections, nothing else:
Definition: ...
Intuition: ...
Example: ...
Key points: ...
Never assume advanced knowledge for a beginner.
"""


def concept_prompt(
    query: str,
    topic: str,
    level: str,
    language: str,
    history: Sequence[Mapping[str, str]] | None = None,
) -> str:
    """Build the user message for the explain_concept node."""
    transcript = ""
    if history:
        transcript = "\n".join(
            f"{item.get('role', 'user')}: {item.get('content', '')}" for item in list(history)[-4:]
        )
    return (
        f"Topic: {topic}\nStudent level: {level}\nAnswer language: {language}\n"
        f"Conversation so far:\n{transcript}\n\n"
        f"Student message:\n{query}\n\n"
        "Explain the concept."
    )


SOLVE_SYSTEM_PROMPT = """\
You are a mathematics tutor solving a problem step by step.
Never give a bare answer: always justify every step.
Answer with these labelled sections, nothing else:
Analysis: what is given and what is asked
Method: which rule or formula is used and why
Steps: numbered steps, each showing the arithmetic
Final answer: the single final value
Keep the language and difficulty aligned with the student's level.
If the problem cannot be solved with the given data, say so in Analysis.
"""


def solve_prompt(
    query: str,
    problem_statement: str,
    level: str,
    language: str,
    previous_attempt: Mapping[str, Any] | None = None,
    feedback: Sequence[str] | None = None,
) -> str:
    """Build the user message for solve_problem / repair_solution."""
    parts = [
        f"Student level: {level}",
        f"Answer language: {language}",
        f"Problem as extracted by the router: {problem_statement or query}",
        f"Student message:\n{query}",
    ]
    if previous_attempt:
        parts.append(
            "Previous attempt that failed verification:\n"
            f"{previous_attempt.get('candidate_answer')!r}\n"
            f"steps: {previous_attempt.get('solution_steps')}"
        )
    if feedback:
        parts.append("Checker feedback to fix:\n" + "\n".join(f"- {item}" for item in feedback))
        parts.append("Produce a corrected full solution.")
    parts.append("Return the labelled sections only.")
    return "\n\n".join(parts)


REVIEW_SYSTEM_PROMPT = """\
You are an independent mathematical reviewer. You did NOT produce the solution.
Check the arithmetic, the signs, the units and the final answer.
Answer ONLY with a JSON object:
{"is_correct": true|false, "notes": ["..."], "corrected_answer": <number or null>}
Be strict: recompute every arithmetic line yourself.
"""


def review_prompt(
    problem_statement: str,
    steps: Sequence[str],
    candidate_answer: Any,
) -> str:
    """Build the user message for the model based answer review."""
    numbered = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(steps or []))
    return (
        f"Problem:\n{problem_statement}\n\n"
        f"Proposed steps:\n{numbered}\n\n"
        f"Proposed final answer: {candidate_answer!r}\n\n"
        "Verify and return the JSON object."
    )


IMAGE_SYSTEM_PROMPT = """\
You are a vision assistant for a mathematics tutor. You are given an image of a
mathematics problem. Transcribe the problem exactly (keep numbers, signs and
symbols), state the topic, and say whether the image is legible. Do not solve
it. Reply with a short plain-text transcription.
"""


def image_prompt(question: str | None = None) -> str:
    """Build the user message for the inspect_image node."""
    base = "Transcribe the mathematics problem shown in this image."
    if question:
        return f"{base}\nAdditional context from the student: {question}"
    return base


SUMMARY_SYSTEM_PROMPT = """\
You are a mathematics tutor writing the final answer for a student.
Use the verified material you are given. Do not invent new computations.
Respect the student's level and language. Be warm but concise.
"""


def summary_prompt(
    request_type: str,
    level: str,
    language: str,
    solution: Mapping[str, Any],
) -> str:
    """Build the user message for the format_final_answer node."""
    return (
        f"Request type: {request_type}\nStudent level: {level}\nAnswer language: {language}\n\n"
        f"Verified solution material (JSON):\n{solution}\n\n"
        "Write the final answer for the student using the required sections."
    )
