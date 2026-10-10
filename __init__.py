"""Complex Mathematics Tutor Agent.

A graph based CLI agent that classifies a mathematics request, routes it
through the appropriate branch of the graph (concept explanation, step-by-step
solution, or a clarification loop) and verifies the answer before showing it.

Modules
-------
``agent``          the graph, its state, nodes, routing and the ``TutorAgent``
``cli``            interactive terminal front-end
``config``         environment / .env based configuration
``llm_provider``   provider abstraction (real model + offline fallback)
``heuristics``     deterministic mathematics engine and independent verifier
``prompts``        every prompt template used by the graph
"""

from __future__ import annotations

__all__ = [
    "TutorAgent",
    "TutorState",
    "Config",
    "build_provider",
    "LANGGRAPH_AVAILABLE",
    "__version__",
]

__version__ = "1.0.0"


def __getattr__(name: str):  # pragma: no cover - trivial lazy import
    """Lazily expose the main classes so ``import complex_math_tutor`` stays cheap."""
    if name in {"TutorAgent", "TutorState", "LANGGRAPH_AVAILABLE"}:
        from . import agent as _agent

        return getattr(_agent, name)
    if name == "Config":
        from .config import Config as _Config

        return _Config
    if name == "build_provider":
        from .llm_provider import build_provider as _build_provider

        return _build_provider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
