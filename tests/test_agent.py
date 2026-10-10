"""Tests for the agent graph: state, nodes, routing and verification.

Everything here runs against the offline deterministic provider, so the suite
needs no API key and no network access.
"""

from __future__ import annotations

import pytest

from complex_math_tutor import heuristics as h
from complex_math_tutor.agent import (
    MAX_TOTAL_STEPS,
    NODES,
    AgentDeps,
    TutorAgent,
    TutorState,
    check_problem_clarity,
    classify_request,
    next_node,
    read_problem,
)
from complex_math_tutor.config import Config
from fakes import FailingProvider, FakeProvider

# --------------------------------------------------------------------------- #
# Graph structure
# --------------------------------------------------------------------------- #

def test_all_reference_nodes_exist() -> None:
    """Every node named in the reference diagram is registered."""
    expected = {
        "read_problem",
        "inspect_image",
        "classify_request",
        "check_problem_clarity",
        "ask_question",
        "explain_concept",
        "solve_problem",
        "check_answer",
        "repair_solution",
        "format_final_answer",
        "final_answer",
    }
    assert expected.issubset(set(NODES))


def test_next_node_is_a_real_router(offline_agent: TutorAgent) -> None:
    """The router branches on state instead of following a fixed line."""
    deps = offline_agent.deps
    unclear: TutorState = {"is_problem_clear": False, "iteration_count": 0}
    clear_concept: TutorState = {"is_problem_clear": True, "route_decision": "explain_concept"}
    clear_solve: TutorState = {"is_problem_clear": True, "route_decision": "solve_problem"}
    wrong: TutorState = {"is_answer_correct": False, "solve_attempts": 1}
    right: TutorState = {"is_answer_correct": True, "solve_attempts": 1}

    assert next_node("check_problem_clarity", unclear, deps) == "ask_question"
    assert next_node("check_problem_clarity", clear_concept, deps) == "explain_concept"
    assert next_node("check_problem_clarity", clear_solve, deps) == "solve_problem"
    assert next_node("check_answer", wrong, deps) == "repair_solution"
    assert next_node("check_answer", right, deps) == "format_final_answer"


def test_clarification_loop_is_bounded(offline_agent: TutorAgent) -> None:
    """Once the clarification budget is spent the graph stops looping."""
    deps = offline_agent.deps
    exhausted: TutorState = {
        "is_problem_clear": False,
        "iteration_count": offline_agent.config.max_clarification_rounds,
    }
    assert next_node("check_problem_clarity", exhausted, deps) == "final_answer"


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "query,expected_type",
    [
        ("مشتق تابع x^2 + 3x را توضیح بده", "mixed"),
        ("مشتق چیست؟", "conceptual"),
        ("اگر x=2 باشد، مقدار 3x^2 + 1 چقدر است؟", "solvable_problem"),
        ("2x + 3 = 7", "solvable_problem"),
        ("سلام", "unclear"),
    ],
)
def test_request_type_detection(offline_agent: TutorAgent, query: str, expected_type: str) -> None:
    """Conceptual, solvable, mixed and unclear requests are told apart."""
    state = offline_agent.run(query)
    assert state["request_type"] == expected_type


def test_concept_question_does_not_enter_the_numeric_path(offline_agent: TutorAgent) -> None:
    """A pure concept question is explained without a numeric solve."""
    state = offline_agent.run("مشتق چیست؟")
    assert state["request_type"] == "conceptual"
    assert "explain_concept" in state["path_log"]
    assert "solve_problem" not in state["path_log"]
    assert state["concept_sections"]
    assert "تعریف" in state["final_answer"]


def test_solvable_problem_produces_a_verified_numeric_answer(offline_agent: TutorAgent) -> None:
    """A substitution problem is solved, checked and reported correctly."""
    state = offline_agent.run("اگر x=2 باشد، مقدار 3x^2 + 1 چقدر است؟")
    assert state["request_type"] == "solvable_problem"
    assert state["checked_answer"] == pytest.approx(13.0)
    assert state["is_answer_correct"] is True
    assert state["verification_method"] in {"recomputation", "structural"}
    assert "13" in state["final_answer"]


def test_topic_and_level_are_extracted(offline_agent: TutorAgent) -> None:
    """Topic and an explicit user level are detected and stored."""
    state = offline_agent.run("من مبتدی هستم. انتگرال چیست؟")
    assert state["subject_or_topic"] == "integral"
    assert state["user_level"] == "beginner"


def test_unknown_level_defaults_to_beginner(offline_agent: TutorAgent) -> None:
    """With no level information the explanation is pitched at beginners."""
    state = offline_agent.run("مشتق چیست؟")
    assert state["user_level"] == "beginner"


# --------------------------------------------------------------------------- #
# Clarity / clarification loop
# --------------------------------------------------------------------------- #

def test_incomplete_question_triggers_ask_question(offline_agent: TutorAgent) -> None:
    """An unanswerable request routes to ask_question and stops for input."""
    state = offline_agent.run("سلام، حالت چطوره؟")
    assert state["is_problem_clear"] is False
    assert "ask_question" in state["path_log"]
    assert state["awaiting_input"] is True
    assert state["clarification_question"]
    assert state["final_answer"] == state["clarification_question"]


def test_missing_variable_value_triggers_clarification(offline_agent: TutorAgent) -> None:
    """A pinned x with an expression that still needs y is a clarification."""
    state = offline_agent.run("حل کن x=2 3y+1")
    assert state["is_problem_clear"] is False
    assert state["missing_information"] == "no_variable_value"
    assert "ask_question" in state["path_log"]


def test_scripted_clarification_answer_resumes_the_graph(scripted_agent) -> None:
    """ask_question -> read_problem -> ... completes when the user answers."""
    agent = scripted_agent(["اگر x=2 باشد، مقدار 3x^2 + 1 چقدر است؟"])
    state = agent.run("سلام")
    assert state["iteration_count"] == 1
    assert state["path_log"].count("read_problem") == 2
    assert state["checked_answer"] == pytest.approx(13.0)
    assert state["final_answer"].count("13") >= 1


def test_clarification_loop_terminates_without_an_answer(scripted_agent) -> None:
    """An endlessly unclear conversation still terminates."""
    agent = scripted_agent(["", "", "", "", ""])
    state = agent.run("سلام")
    assert len(state["path_log"]) <= MAX_TOTAL_STEPS
    assert state["path_log"][-1] == "final_answer"
    assert state["final_answer"]


# --------------------------------------------------------------------------- #
# State updates
# --------------------------------------------------------------------------- #

def test_state_is_updated_throughout_the_run(offline_agent: TutorAgent) -> None:
    """The state really changes while the graph executes."""
    query = "اگر x=2 باشد، مقدار 3x^2 + 1 چقدر است؟"
    initial = offline_agent.create_initial_state(query)
    # a fresh state carries the raw input but none of the derived fields
    assert initial["solution_steps"] == []
    assert initial["final_answer"] == ""
    assert initial["problem_statement"] == ""
    assert initial["is_problem_clear"] is False
    assert initial["request_type"] == "unclear"

    final = offline_agent.run(query)
    assert final["normalized_query"]
    assert final["problem_statement"]
    assert final["solution_steps"]
    assert final["checked_answer"] is not None
    assert final["final_answer"]
    assert final["conversation_history"]
    assert final["iteration_count"] == 0
    assert final["path_log"]
    # the derived fields really changed
    assert final["problem_statement"] != initial["problem_statement"]
    assert final["request_type"] != initial["request_type"]
    assert final["is_problem_clear"] is True


def test_read_problem_normalises_and_records(offline_agent: TutorAgent) -> None:
    """read_problem cleans the text, detects topic/level and logs history."""
    deps = offline_agent.deps
    state = offline_agent.create_initial_state("من مبتدی هستم: مشتق ۳x^۲ را توضیح بده")
    updates = read_problem(state, deps)
    assert updates["normalized_query"].startswith("من مبتدی هستم")
    assert "3x^2" in updates["normalized_query"]  # Persian digits normalised
    assert updates["subject_or_topic"] == "derivative"
    assert updates["user_level"] == "beginner"
    assert updates["conversation_history"][-1]["role"] == "user"


def test_classify_request_returns_routing_metadata(offline_agent: TutorAgent) -> None:
    """classify_request fills every routing field of the state."""
    deps = offline_agent.deps
    state = offline_agent.create_initial_state("2x + 3 = 7")
    state.update(read_problem(state, deps))
    updates = classify_request(state, deps)
    for key in (
        "request_type",
        "subject_or_topic",
        "user_level",
        "problem_statement",
        "is_problem_clear",
        "missing_information",
    ):
        assert key in updates
    assert updates["request_type"] == "solvable_problem"


def test_check_problem_clarity_creates_a_real_decision(offline_agent: TutorAgent) -> None:
    """The clarity node forces a route decision instead of being decorative."""
    deps = offline_agent.deps
    unclear: TutorState = {"request_type": "unclear", "language": "fa"}
    updates = check_problem_clarity(unclear, deps)
    assert updates["is_problem_clear"] is False
    assert updates["route_decision"] == "ask_question"
    assert updates["clarification_question"]

    solvable: TutorState = {
        "request_type": "solvable_problem",
        "problem_statement": "2*x+3=7",
        "language": "fa",
        "is_problem_clear": True,
    }
    updates = check_problem_clarity(solvable, deps)
    assert updates["is_problem_clear"] is True
    assert updates["route_decision"] == "solve_problem"


def test_conversation_context_is_kept_between_turns(offline_agent: TutorAgent) -> None:
    """A resumed run keeps the previous conversation history."""
    first = offline_agent.run("مشتق چیست؟")
    second = offline_agent.run("مثال بزن", state=first)
    assert len(second["conversation_history"]) > len(first["conversation_history"])


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #

def test_check_answer_records_a_verdict(offline_agent: TutorAgent) -> None:
    """is_answer_correct is really written into the state."""
    good = offline_agent.run("اگر x=2 باشد، مقدار 3x^2 + 1 چقدر است؟")
    assert good["is_answer_correct"] is True
    assert good["verification_notes"]
    assert good["verification_method"]


def test_wrong_answer_is_repaired_then_rechecked() -> None:
    """A failing check sends the graph through repair_solution."""
    provider = FakeProvider(
        classify_result={
            "request_type": "solvable_problem",
            "problem_statement": "3x^2 + 1",
            "is_clear": True,
            "subject_or_topic": "arithmetic",
            "user_level": "beginner",
        },
        solve_result={"steps": ["1. محاسبه"], "answer": 99.0, "method": "fake"},
        review_result={"is_correct": False, "notes": ["پاسخ اشتباه است"], "corrected_answer": 13.0},
    )
    agent = TutorAgent(config=Config.from_env(provider="deterministic", api_key=None), provider=provider)
    state = agent.run("اگر x=2 باشد، مقدار 3x^2 + 1 چقدر است؟")

    assert "repair_solution" in state["path_log"]
    assert state["path_log"].count("check_answer") == 2
    assert state["solve_attempts"] == 2
    assert state["is_answer_correct"] is False
    assert state["verification_notes"]


def test_verifier_rejects_a_wrong_numeric_answer() -> None:
    """The independent recomputation catches an incorrect answer."""
    verdict = h.verify_answer(["x = 3*4 + 1 = 13"], 99.0, "3*x**2+1", {"x": 2})
    assert verdict["is_correct"] is False
    assert verdict["method"] == "recomputation"


def test_verifier_accepts_a_correct_numeric_answer() -> None:
    """A correct answer passes the independent recomputation."""
    verdict = h.verify_answer(["x = 3*4 + 1 = 13"], 13.0, "3*x**2+1", {"x": 2})
    assert verdict["is_correct"] is True


def test_equation_roots_are_verified_by_substitution() -> None:
    """Equation answers are checked by plugging the roots back in."""
    verdict = h.verify_answer([], 2.0, "2*x+3=7", {}, "equation")
    assert verdict["is_correct"] is True
    assert verdict["method"] == "root_substitution"

    bad = h.verify_answer([], 5.0, "2*x+3=7", {}, "equation")
    assert bad["is_correct"] is False


def test_derivative_is_cross_checked_numerically(offline_agent: TutorAgent) -> None:
    """The derivative answer is verified with a finite difference."""
    state = offline_agent.run("مشتق x^2 + 3x را حل کن و جواب بده")
    assert state["is_answer_correct"] is True
    assert "2x + 3" in str(state["checked_answer"])
    assert any("تفاضل عددی" in note for note in state["verification_notes"])


def test_solver_failure_is_reported_without_crashing() -> None:
    """A provider outage degrades gracefully into an error message."""
    agent = TutorAgent(
        config=Config.from_env(provider="deterministic", api_key=None),
        provider=FailingProvider(
            classify_result={
                "request_type": "solvable_problem",
                "problem_statement": "2x + 3 = 7",
                "is_clear": True,
            }
        ),
    )
    state = agent.run("2x + 3 = 7")
    assert state["error_message"]
    assert state["final_answer"]
    assert state["path_log"][-1] == "final_answer"


# --------------------------------------------------------------------------- #
# Backends and configuration
# --------------------------------------------------------------------------- #

def test_agent_reports_its_backend(offline_agent: TutorAgent) -> None:
    """The agent always exposes which execution backend is active."""
    assert offline_agent.backend in {"langgraph", "builtin"}
    assert "graph backend" in offline_agent.describe()


def test_run_is_deterministic_for_the_offline_provider(offline_agent: TutorAgent) -> None:
    """Two identical runs produce identical answers."""
    first = offline_agent.run("اگر x=2 باشد، مقدار 3x^2 + 1 چقدر است؟")
    second = offline_agent.run("اگر x=2 باشد، مقدار 3x^2 + 1 چقدر است؟")
    assert first["final_answer"] == second["final_answer"]
    assert first["path_log"] == second["path_log"]


def test_run_turns_keeps_context(offline_agent: TutorAgent) -> None:
    """run_turns walks several turns sharing one conversation."""
    states = offline_agent.run_turns(["مشتق چیست؟", "اگر x=3 باشد مقدار x^2 چقدر است؟"])
    assert len(states) == 2
    assert states[1]["checked_answer"] == pytest.approx(9.0)


def test_agent_deps_bundle_is_exposed(offline_agent: TutorAgent) -> None:
    """AgentDeps can be built from the agent (used by the CLI and tests)."""
    deps = offline_agent.deps
    assert isinstance(deps, AgentDeps)
    assert deps.config is offline_agent.config
    assert deps.provider is offline_agent.provider
