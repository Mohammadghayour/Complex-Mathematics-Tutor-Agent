"""Interactive command line interface for the Complex Mathematics Tutor Agent.

This module contains **only** presentation logic: argument parsing, terminal
I/O, command handling and error rendering.  All agent behaviour lives in
:mod:`complex_math_tutor.agent`.

Usage
-----
::

    python cli.py                      # interactive session
    python cli.py --debug              # interactive session with trace logging
    python cli.py --once "مشتق x^2 چیست؟"
    python cli.py --image images/q.png # analyse an image problem
    python -m complex_math_tutor.cli --provider openai --model gpt-4o-mini

Commands available inside the session::

    exit | quit | :q        leave
    help | :h               show help
    reset | new             forget the current conversation context
    image                   provide a path to an image of the problem
    level <beginner|intermediate|advanced>   pin the user level
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

# --------------------------------------------------------------------------- #
# Import bootstrap: allow both `python cli.py` and `python -m complex_math_tutor.cli`
# --------------------------------------------------------------------------- #
if __package__ in (None, ""):  # executed as a plain script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from complex_math_tutor import heuristics as h  # noqa: E402
from complex_math_tutor.agent import (  # noqa: E402
    LANGGRAPH_AVAILABLE,
    TutorAgent,
    TutorState,
    resolve_image_path,
    setup_logging,
    validate_image_file,
)
from complex_math_tutor.config import KNOWN_PROVIDERS, Config  # noqa: E402
from complex_math_tutor.llm_provider import LLMError  # noqa: E402

__version__ = "1.0.0"

BANNER = """\
=========================================================
  Complex Mathematics Tutor Agent  (v{version})
=========================================================
  Type your mathematics question and press Enter.
  Commands: exit | quit | help | reset | image | level
  Backend : {backend}
  Provider: {provider}
  Vision  : {vision}
  Images  : {images_dir}
========================================================="""

HELP_TEXT = """\
Commands
  <question>              ask a maths question (conceptual or solvable)
  image                   give the path of an image containing the problem
  level beginner|intermediate|advanced
                          pin the explanation level (default: auto-detected)
  reset                   start a new conversation (clears the context)
  help                    show this help
  exit | quit | :q        quit (Ctrl+C and Ctrl+D also work)

Notes
  * Put your image inside the project images/ folder and type its path, e.g.
        image
        Image path: images/math_question.png
  * Relative and absolute paths are both accepted.
"""


# --------------------------------------------------------------------------- #
# Terminal helpers
# --------------------------------------------------------------------------- #

def configure_stdio() -> None:
    """Make Persian/English output safe on Windows consoles and pipes."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - exotic streams
                pass


def build_input_reader(prompt: str = "You: ") -> Callable[[str], str]:
    """Return a reader that uses ``prompt_toolkit`` when available.

    Falls back to the built-in :func:`input`, which keeps the CLI dependency
    free while still offering history and line editing when prompt_toolkit is
    installed.
    """
    try:  # pragma: no cover - depends on the environment
        from prompt_toolkit import PromptSession  # type: ignore
        from prompt_toolkit.history import InMemoryHistory  # type: ignore

        session: Any = PromptSession(history=InMemoryHistory())

        def reader(_: str = prompt) -> str:
            return session.prompt(prompt)

        return reader
    except Exception:  # pragma: no cover - prompt_toolkit missing or broken

        def reader(_: str = prompt) -> str:
            return input(prompt)

        return reader


def safe_print(text: str = "") -> None:
    """Print without ever raising on encoding problems."""
    try:
        print(text)
    except UnicodeEncodeError:  # pragma: no cover - last resort
        print(text.encode("ascii", errors="replace").decode("ascii"))


def render_turn(agent: TutorAgent, state: Mapping[str, Any], debug: bool = False) -> None:
    """Print the header plus the final answer of one completed turn."""
    safe_print("")
    safe_print(agent.describe_turn(state))
    if debug:
        path = " -> ".join(state.get("path_log") or [])
        safe_print(f"[debug] مسیر اجرای گراف: {path}")
        safe_print(
            f"[debug] iteration={state.get('iteration_count')} "
            f"solve_attempts={state.get('solve_attempts')} "
            f"verified={state.get('is_answer_correct')} "
            f"method={state.get('verification_method')}"
        )
    if state.get("error_message"):
        safe_print(f"[هشدار / warning] {state['error_message']}")
    safe_print("")
    safe_print(str(state.get("final_answer") or "").strip())
    safe_print("")


def resolve_level_argument(value: str | None, current: str | None) -> str | None:
    """Validate a level given with the ``level`` command."""
    if value is None:
        return current
    cleaned = value.strip().lower()
    if cleaned in {"beginner", "intermediate", "advanced", "auto"}:
        return None if cleaned == "auto" else cleaned
    safe_print("سطح نامعتبر است. مقادیر مجاز: beginner | intermediate | advanced | auto")
    return current


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="complex-math-tutor",
        description="CLI agent that explains mathematics concepts and solves problems step by step.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python cli.py\n"
            "  python cli.py --debug --once \"مشتق x^2 + 3x را توضیح بده\"\n"
            "  python cli.py --image images/math_question.png\n"
            "  python cli.py --provider openai --model gpt-4o-mini\n"
        ),
    )
    parser.add_argument("--model", help="language model name (overrides MATH_TUTOR_MODEL)")
    parser.add_argument(
        "--provider",
        choices=KNOWN_PROVIDERS,
        help="LLM provider; 'auto' uses a real model when an API key exists, else the offline engine",
    )
    parser.add_argument("--image", help="path to an image of the problem (PNG/JPG/WEBP/GIF/BMP)")
    parser.add_argument("--debug", action="store_true", help="show the graph path and verbose logging")
    parser.add_argument("--env", help="path to a .env file (default: complex_math_tutor/.env)")
    parser.add_argument("--once", metavar="QUERY", help="answer a single query and exit (no interactive loop)")
    parser.add_argument(
        "--no-model-review",
        action="store_true",
        help="skip the extra model based answer review (deterministic verification always runs)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="kept for compatibility; the CLI never emits ANSI colour",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    """Merge CLI arguments into the environment derived configuration."""
    overrides: dict[str, Any] = {
        "debug": bool(args.debug),
        "model": args.model,
        "provider": args.provider,
    }
    if args.no_model_review:
        overrides["allow_model_review"] = False
    config = Config.from_env(env_file=args.env, **overrides)
    if args.debug:
        setup_logging(debug=True)
    return config


# --------------------------------------------------------------------------- #
# Session
# --------------------------------------------------------------------------- #

class TutorCLI:
    """Interactive terminal front-end for :class:`TutorAgent`."""

    def __init__(self, agent: TutorAgent, debug: bool = False) -> None:
        self.agent = agent
        self.debug = debug
        self.state: TutorState | None = None
        self.pinned_level: str | None = None
        self.quit_requested = False
        self.reader = build_input_reader()

    # -- session helpers ------------------------------------------------ #
    def show_banner(self) -> None:
        """Print the startup banner."""
        config = self.agent.config
        safe_print(
            BANNER.format(
                version=__version__,
                backend=self.agent.backend,
                provider=f"{self.agent.provider_name} / {config.model if config.has_language_model else 'offline engine'}",
                vision="enabled" if getattr(self.agent.provider, "supports_vision", False) else "disabled (no vision model)",
                images_dir=str(config.images_dir),
            )
        )
        if not LANGGRAPH_AVAILABLE:
            safe_print(
                "توجه: LangGraph نصب نیست؛ از اجراکننده داخلی استفاده می‌شود "
                "(pip install -r requirements.txt).\n"
            )

    def run_query(self, query: str, image_path: str | None = None) -> TutorState:
        """Execute one turn through the agent and render the result."""
        try:
            state = self.agent.run(
                query,
                image_path=image_path,
                input_callback=self.ask_follow_up,
                state=self.state,
            )
        except LLMError as exc:
            safe_print(f"\n[خطای مدل زبانی / model error] {exc}\n")
            return self.state or self.agent.create_initial_state(query, image_path)
        except KeyboardInterrupt:  # pragma: no cover - interactive
            safe_print("\n[لغو شد / cancelled]\n")
            return self.state or self.agent.create_initial_state(query, image_path)
        except Exception as exc:  # pragma: no cover - defensive
            logging.getLogger("complex_math_tutor.cli").exception("unexpected failure")
            safe_print(f"\n[خطای غیرمنتظره / unexpected error] {exc}")
            safe_print("اگر مشکل ادامه داشت با --debug اجرا کنید تا جزئیات را ببینید.\n")
            return self.state or self.agent.create_initial_state(query, image_path)

        self.state = state
        render_turn(self.agent, state, self.debug)
        if state.get("awaiting_input"):
            if self.quit_requested:
                safe_print("برای خروج exit را بزنید یا Ctrl+C. / Type exit or press Ctrl+C to leave.")
            else:
                safe_print("برای ادامه، پاسخ سؤال بالا را وارد کنید.")
        return state

    def ask_follow_up(self, question: str) -> str:
        """Callback used by the graph's ``ask_question`` node.

        Typing ``exit``/``quit`` at the follow-up prompt ends the session
        instead of being sent to the agent as an answer.
        """
        safe_print("")
        safe_print(f"Agent: {question}")
        try:
            answer = self.reader("Answer: ")
        except (EOFError, KeyboardInterrupt):
            return ""
        if (answer or "").strip().lower() in {"exit", "quit", ":q", ":quit", ":exit"}:
            self.quit_requested = True
            return ""
        return answer

    # -- commands ------------------------------------------------------- #
    def handle_image_command(self, inline_path: str | None = None) -> None:
        """Ask for an image path, validate it and run the agent on it."""
        raw = inline_path
        if not raw:
            try:
                raw = self.reader("Image path: ")
            except (EOFError, KeyboardInterrupt):
                safe_print("\nلغو شد.")
                return
        raw = (raw or "").strip().strip('"').strip("'")
        if not raw:
            safe_print("مسیری وارد نشد.")
            return

        resolved = resolve_image_path(raw, self.agent.config)
        error = validate_image_file(resolved)
        if error:
            safe_print(f"[خطای تصویر / image error] {error}")
            safe_print(
                f"تصویر را در پوشه {self.agent.config.images_dir} قرار دهید و مسیر آن را وارد کنید "
                "(مسیر نسبی یا مطلق هر دو مجاز است)."
            )
            if not getattr(self.agent.provider, "supports_vision", False):
                safe_print(
                    "مدل بینایی در دسترس نیست؛ برای فعال‌سازی OPENAI_API_KEY را تنظیم کنید "
                    "یا مسئله را به صورت متنی بنویسید."
                )
            return

        safe_print(f"تصویر دریافت شد و در حال تحلیل است... ({resolved})")
        try:
            extra = self.reader("سؤال متنی (اختیاری، Enter برای رد کردن): ")
        except (EOFError, KeyboardInterrupt):
            extra = ""
        self.run_query((extra or "").strip(), image_path=str(resolved))

    def handle_level_command(self, argument: str | None) -> None:
        """Pin or clear the user level used for explanations."""
        self.pinned_level = resolve_level_argument(argument, self.pinned_level)
        if self.state is not None and self.pinned_level:
            self.state["user_level"] = self.pinned_level  # type: ignore[index]
        safe_print(f"سطح فعلی: {self.pinned_level or 'تشخیص خودکار'}")

    def handle_command(self, raw: str) -> bool:
        """Handle a meta command. Returns ``True`` when the session should stop."""
        command = raw.strip()
        lowered = command.lower()

        if lowered in {"exit", "quit", ":q", ":quit", ":exit"}:
            safe_print("خدانگهدار! / Goodbye!")
            return True
        if lowered in {"help", ":h", "?"}:
            safe_print(HELP_TEXT)
            return False
        if lowered in {"reset", "new", ":reset"}:
            self.state = None
            safe_print("زمینه گفتگو پاک شد. / Conversation context cleared.")
            return False
        if lowered == "image" or lowered.startswith("image "):
            inline = command[len("image"):].strip() or None
            self.handle_image_command(inline)
            return False
        if lowered.startswith("level"):
            argument = command[len("level"):].strip() or None
            self.handle_level_command(argument)
            return False
        safe_print(f"دستور ناشناخته: {command} (برای راهنما help را بنویسید)")
        return False

    # -- main loop ------------------------------------------------------ #
    def run(self, initial_image: str | None = None, once: str | None = None) -> int:
        """Run the CLI. Returns the process exit code."""
        if once:
            if initial_image:
                resolved = resolve_image_path(initial_image, self.agent.config)
                error = validate_image_file(resolved)
                if error:
                    safe_print(f"[خطای تصویر / image error] {error}")
                    return 2
                self.run_query(once, image_path=str(resolved))
            else:
                self.run_query(once)
            return 0

        self.show_banner()
        if initial_image:
            self.handle_image_command(initial_image)

        while True:
            try:
                raw = self.reader()
            except (EOFError, KeyboardInterrupt):
                safe_print("\nخدانگهدار! / Goodbye!")
                return 0
            except Exception as exc:  # pragma: no cover - defensive
                safe_print(f"[خطای خواندن ورودی / input error] {exc}")
                return 1

            text = (raw or "").strip()
            if not text:
                continue
            if self.quit_requested:
                safe_print("خدانگهدار! / Goodbye!")
                return 0
            if text.startswith(("exit", "quit", "help", "reset", "new", "image", "level", ":q", ":h", "?")):
                if self.handle_command(text):
                    return 0
                continue
            if text.startswith(":"):
                if self.handle_command(text):
                    return 0
                continue

            query = text
            if self.pinned_level:
                query = f"[سطح کاربر: {self.pinned_level}] {query}"
            self.run_query(query)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)

    try:
        agent = TutorAgent(config=config)
    except Exception as exc:  # pragma: no cover - defensive
        safe_print(f"[خطای راه‌اندازی / startup error] {exc}")
        return 1

    cli = TutorCLI(agent, debug=bool(args.debug))
    return cli.run(initial_image=args.image, once=args.once)


if __name__ == "__main__":  # pragma: no cover - manual entry
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        safe_print("\nخدانگهدار! / Goodbye!")
        raise SystemExit(0)
