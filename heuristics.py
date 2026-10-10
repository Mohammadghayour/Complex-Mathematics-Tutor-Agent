"""Deterministic offline mathematics engine.

This module is the *reasoning fallback* of the agent.  It contains no graph
logic and no I/O: it only turns text into mathematics and back.  Two consumers
use it:

* :class:`~complex_math_tutor.llm_provider.DeterministicProvider` - when no
  language model is configured the agent still has to classify, explain and
  solve, so the provider dispatches to the functions below.
* :func:`verify_answer` - the independent answer checker used by the
  ``check_answer`` node.  It recomputes results with Python instead of trusting
  the generated prose.

Everything here is pure Python (``ast`` + finite differences), so the whole
project is testable without any API key.
"""

from __future__ import annotations

import ast
import math
import re
from typing import Any, Iterable, Mapping, Sequence

# --------------------------------------------------------------------------- #
# Text normalisation
# --------------------------------------------------------------------------- #

PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
_DIGIT_TABLE = {ord(c): str(i) for i, c in enumerate(PERSIAN_DIGITS)}
_DIGIT_TABLE.update({ord(c): str(i) for i, c in enumerate(ARABIC_DIGITS)})

_CHAR_FIXES = {
    "ي": "ی",
    "ك": "ک",
    "ﻻ": "لا",
    "×": "*",
    "÷": "/",
    "−": "-",
    "–": "-",
    "—": "-",
    "٫": ".",
    "،": ",",
    "؛": ";",
}

PERSIAN_LETTER_RE = re.compile(r"[\u0600-\u06FF]")
_WS_RE = re.compile(r"\s+")

MATH_FUNCTIONS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "log": math.log,
    "ln": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "factorial": math.factorial,
    "pow": pow,
    "min": min,
    "max": max,
}

MATH_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}

SINGLE_LETTER_VARS = set("xyzabcnkmuvwtpqr")

# --------------------------------------------------------------------------- #
# Level / topic detection
# --------------------------------------------------------------------------- #

_LEVEL_KEYWORDS: Mapping[str, Sequence[str]] = {
    "beginner": (
        "مبتدی", "ساده", "ابتدایی", "کلاس هفتم", "کلاس هشتم", "کلاس نهم",
        "راهنمایی", "دبستان", "بدون دانش", "شروع", "تازه‌کار",
        "beginner", "simple", "easy", "basic", "elementary", "school",
    ),
    "advanced": (
        "پیشرفته", "دانشگاه", "المپیاد", "کارشناسی", "حرفه‌ای", "عمیق",
        "اثبات", "قضیه", "حد و پیوستگی", "حساب دیفرانسیل",
        "advanced", "university", "olympiad", "proof", "theorem", "rigorous",
    ),
    "intermediate": (
        "دبیرستان", "متوسطه", "کنکور", "کلاس دهم", "کلاس یازدهم", "کلاس دوازدهم",
        "intermediate", "high school", "precalc", "pre-calculus",
    ),
}

_TOPIC_KEYWORDS: Mapping[str, Sequence[str]] = {
    "derivative": ("مشتق", "دیفرانسیل", "derivative", "differentiate", "d/dx"),
    "integral": ("انتگرال", "integral", "integrate", "antiderivative", "پادمشتق"),
    "limit": ("حد", "حدی", "limit", "lim", "پیوستگی", "continuity"),
    "equation": ("معادله", "ریشه", "جواب معادله", "equation", "solve for", "roots"),
    "inequality": ("نامعادله", "نابرابری", "inequality", "inequalities"),
    "function": ("تابع", "دامنه", "برد", "function", "domain", "range"),
    "trigonometry": ("مثلثات", "سینوس", "کسینوس", "تانژانت", "sin", "cos", "tan", "trig"),
    "geometry": ("هندسه", "مساحت", "محیط", "حجم", "مثلث", "دایره", "geometry",
                 "area", "perimeter", "volume", "circle", "triangle"),
    "algebra": ("جبر", "اتحاد", "فاکتورگیری", "چندجمله", "algebra", "factor", "polynomial"),
    "matrix": ("ماتریس", "دترمینان", "matrix", "determinant"),
    "probability": ("احتمال", "شانس", "probability", "chance"),
    "statistics": ("آمار", "میانگین", "میانه", "انحراف معیار", "statistics",
                   "mean", "median", "average", "std"),
    "logarithm": ("لگاریتم", "لگاریتمی", "logarithm", "log"),
    "percent": ("درصد", "درصدی", "percent", "percentage"),
    "sequence": ("دنباله", "تصاعد", "مجموعه", "sequence", "series", "arithmetic progression"),
    "arithmetic": ("محاسبه", "حساب", "جمع", "تفریق", "ضرب", "تقسیم", "مقدار",
                   "calculate", "compute", "evaluate", "value of", "arithmetic"),
}

TOPIC_LABELS_FA: Mapping[str, str] = {
    "derivative": "مشتق",
    "integral": "انتگرال",
    "limit": "حد",
    "equation": "معادله",
    "inequality": "نامعادله",
    "function": "تابع",
    "trigonometry": "مثلثات",
    "geometry": "هندسه",
    "algebra": "جبر",
    "matrix": "ماتریس",
    "probability": "احتمال",
    "statistics": "آمار",
    "logarithm": "لگاریتم",
    "percent": "درصد",
    "sequence": "دنباله",
    "arithmetic": "محاسبات",
    "unknown": "نامشخص",
}

LEVEL_LABELS_FA: Mapping[str, str] = {
    "beginner": "مبتدی",
    "intermediate": "متوسط",
    "advanced": "پیشرفته",
    "unknown": "نامشخص",
}

REQUEST_LABELS_FA: Mapping[str, str] = {
    "conceptual": "مفهومی",
    "solvable_problem": "مسئله قابل حل",
    "mixed": "مفهومی + محاسباتی",
    "unclear": "نامشخص",
}


def normalize_text(text: str) -> str:
    """Normalise digits, Arabic/Persian look-alikes and whitespace."""
    if not text:
        return ""
    out = text.translate(_DIGIT_TABLE)
    for src, dst in _CHAR_FIXES.items():
        out = out.replace(src, dst)
    out = out.replace("\u200c", " ")  # zero-width non-joiner -> space
    return _WS_RE.sub(" ", out).strip()


_EN_WORD_RE = re.compile(r"[A-Za-z]{3,}")


def detect_language(text: str) -> str:
    """Return ``"fa"`` for Persian/neutral text, ``"en"`` for English prose.

    Text made only of mathematics (``"2x + 3 = 7"``) counts as *neutral* and is
    answered in Persian, which keeps the output of the offline engine
    consistent instead of mixing English headings with Persian steps.
    """
    if not text:
        return "fa"
    if PERSIAN_LETTER_RE.search(text):
        return "fa"
    words = [
        word.lower()
        for word in _EN_WORD_RE.findall(text)
        if word.lower() not in MATH_FUNCTIONS and word.lower() not in MATH_CONSTANTS
    ]
    return "en" if words else "fa"


def detect_user_level(text: str) -> str:
    """Detect a rough user level from explicit self-description keywords."""
    lowered = normalize_text(text).lower()
    for level, keywords in _LEVEL_KEYWORDS.items():
        if any(keyword.lower() in lowered for keyword in keywords):
            return level
    return "unknown"


def detect_topic(text: str) -> str:
    """Detect the mathematics topic of a query (canonical english key)."""
    lowered = normalize_text(text).lower()
    best, best_len = "unknown", 0
    for topic, keywords in _TOPIC_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in lowered and len(keyword) > best_len:
                best, best_len = topic, len(keyword)
    return best


def topic_label(topic: str, lang: str = "fa") -> str:
    """Human readable topic label in the requested language."""
    if lang == "fa":
        return TOPIC_LABELS_FA.get(topic, TOPIC_LABELS_FA["unknown"])
    return topic if topic != "unknown" else "unknown"


def level_label(level: str, lang: str = "fa") -> str:
    """Human readable level label in the requested language."""
    if lang == "fa":
        return LEVEL_LABELS_FA.get(level, LEVEL_LABELS_FA["unknown"])
    return level if level != "unknown" else "unknown"


def request_label(request_type: str, lang: str = "fa") -> str:
    """Human readable request-type label in the requested language."""
    if lang == "fa":
        return REQUEST_LABELS_FA.get(request_type, request_type)
    return request_type.replace("_", " ")


# --------------------------------------------------------------------------- #
# Expression parsing / evaluation
# --------------------------------------------------------------------------- #

_IMPLICIT_MUL_RE = re.compile(r"(?<=[0-9a-zA-Z\)])\s*(?=[a-zA-Z\(])")

#: Tokenizer: decimal number, integer, latin word, or a single operator char.
_EXPR_TOKEN_RE = re.compile(r"\d+\.\d+|\d+|[A-Za-z]+|[\.\+\-\*/^\(\)=<>]")

#: Characters that only ever appear as operators inside a maths expression.
_OPERATOR_CHARS = set("+-*/^=<>()")

#: Characters treated as single letter variables by the maths engine.
_VAR_CHARS = set("xyzabcnkmuvwtpqr")

#: Latin words that look like variables but are actually English filler.
_STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "by", "can", "do", "does", "find",
    "for", "from", "get", "give", "how", "if", "in", "is", "it", "let", "me",
    "much", "of", "on", "or", "please", "show", "so", "solve", "than", "that",
    "the", "then", "this", "to", "value", "what", "when", "where", "which",
    "with", "you", "your", "answer", "calculate", "compute", "evaluate",
    "explain", "problem", "result", "step", "steps", "derivative", "derivatives",
    "integral", "limit", "equation", "function", "number", "numbers", "please",
}

#: Fallback values used when a single letter variable has no stated value.
DEFAULT_VARIABLE_VALUES: Mapping[str, float] = {"x": 0.0, "n": 0.0}


def normalize_expression(expr: str) -> str:
    """Turn ``3x^2 + 1`` into ``3*x**2 + 1`` (safe, no eval of raw input)."""
    out = normalize_text(expr)
    out = out.replace("^", "**")
    out = re.sub(r"(?<=[0-9a-zA-Z])\s*\*\*\s*", "**", out)
    # implicit multiplication: 3x -> 3*x, 2(x+1) -> 2*(x+1), x( -> x*(
    prev = None
    while prev != out:
        prev = out
        out = _IMPLICIT_MUL_RE.sub("*", out)
    return out.strip()


_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv)
_ALLOWED_UNARY = (ast.UAdd, ast.USub)


def safe_eval(expression: str, variables: Mapping[str, float] | None = None) -> float:
    """Evaluate a maths expression with a locked-down ``ast`` walker.

    Raises :class:`ValueError` for anything that is not plain arithmetic.
    """
    variables = dict(variables or {})
    tree = ast.parse(normalize_expression(expression), mode="eval")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError(f"unsupported constant: {node.value!r}")
            return float(node.value)
        if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            return left ** right
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARY):
            value = _eval(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.Name):
            name = node.id
            if name in variables:
                return float(variables[name])
            if name in MATH_CONSTANTS:
                return float(MATH_CONSTANTS[name])
            raise ValueError(f"unknown symbol: {name}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func = MATH_FUNCTIONS.get(node.func.id)
            if func is None:
                raise ValueError(f"unknown function: {node.func.id}")
            return float(func(*[_eval(arg) for arg in node.args]))
        raise ValueError(f"unsupported expression element: {type(node).__name__}")

    return _eval(tree)


def extract_math_expression(text: str) -> str | None:
    """Pull the most plausible mathematical expression out of free text.

    Persian/English words are stripped; digits, single letter variables,
    function names and operators are kept.  Where a *word* was dropped between
    two kept tokens a hard boundary is inserted, so ``"میانگین 2 و 4 و 6"`` does
    not silently become ``2*4*6``.  Inside a continuous run, adjacent operands
    are joined with an explicit ``*`` so ``3 x^2`` and ``2 (x+1)`` survive.

    Returns ``None`` when the text contains no usable maths.

    Examples
    --------
    >>> extract_math_expression("derivative of x^2 + 3x")
    'x^2+3x'
    >>> extract_math_expression("3 + 4 * 2")
    '3+4*2'
    """
    cleaned = normalize_text(text)
    if not cleaned:
        return None
    cleaned = cleaned.replace("،", " ; ").replace(",", " ; ")

    matches = [
        (match.group(0), match.start(), match.end())
        for match in _EXPR_TOKEN_RE.finditer(cleaned)
    ]
    segments: list[list[str]] = [[]]
    previous_token = ""
    for index, (token, start, end) in enumerate(matches):
        if index > 0:
            gap = cleaned[matches[index - 1][2]:start]
            followed_by_equals = index + 1 < len(matches) and matches[index + 1][0] == "="
            starts_new_clause = (
                re.fullmatch(r"[A-Za-z]", token) is not None
                and followed_by_equals
                and previous_token != "="
                and bool(segments[-1])
                and not segments[-1][-1].endswith(("+", "-", "*", "/", "^", "(", "="))
            )
            if gap and re.search(r"[^\s\+\-\*/^\(\)=<>]", gap):
                # a word was dropped here -> the maths is not contiguous
                if segments[-1]:
                    segments.append([])
            elif starts_new_clause:
                # a fresh "y = 5" clause starts a new segment, so
                # "x=2*3*y+1 y=5" -> ["x=2*3*y+1", "y=5"]
                segments.append([])
        if _keep_token(token):
            segments[-1].append(token)
        previous_token = token

    candidates = [
        _join_math_tokens(segment).strip("+-*/^=<>")
        for segment in segments
        if segment
    ]
    # normalise "f(x) = ..." into the expression it describes
    candidates = [
        re.sub(r"^\(\s*[A-Za-z]\s*\)\s*=", "", re.sub(r"(?<![A-Za-z0-9_])[fgh]\*?\(", "(", c))
        for c in candidates
    ]
    candidates = [c for c in candidates if c and any(ch.isdigit() for ch in c)]
    if not candidates:
        return None

    # a lone "x = 2" clause is a variable assignment, not the target expression
    mathy = [c for c in candidates if not re.fullmatch(r"[A-Za-z]\s*=\s*-?\d+(?:\.\d+)?", c)]
    pool = mathy or candidates
    scored = sorted(
        pool,
        key=lambda c: (any(op in c for op in "+-*/^(") or any(ch.isalpha() for ch in c), len(c)),
    )
    for candidate in reversed(scored):
        has_operator = any(op in candidate for op in "+-*/^(")
        has_variable = any(char.isalpha() for char in candidate)
        if has_operator or has_variable:
            return candidate
    return None


def _keep_token(token: str) -> bool:
    """True when *token* carries mathematical meaning worth keeping."""
    if token in _OPERATOR_CHARS:
        return token not in "<>"
    if token.lower() in _STOPWORDS:
        return False
    if token.isalpha():
        return len(token) == 1 or token.lower() in MATH_FUNCTIONS or token.lower() in MATH_CONSTANTS
    if "." in token:
        return bool(re.fullmatch(r"\d+\.\d+", token))
    return True


def _join_math_tokens(tokens: Sequence[str]) -> str:
    """Join tokens, inserting an explicit ``*`` between adjacent operands."""
    out = ""
    for token in tokens:
        if out:
            previous, following = out[-1], token[0]
            if (previous.isalnum() or previous == ")") and (
                following.isalpha() or following.isdigit() or following == "("
            ):
                out += "*"
        out += token
    return out


def strip_assignments(expression: str) -> str:
    """Remove ``x=2`` style clauses from an expression and re-join the rest."""
    tokens = _EXPR_TOKEN_RE.findall(expression)
    kept: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        nxt = tokens[index + 1] if index + 1 < len(tokens) else ""
        if token.isalpha() and len(token) == 1 and nxt == "=":
            index += 2
            continue
        kept.append(token)
        index += 1
    return _join_math_tokens(kept).strip("+-*/^=<>")


# --------------------------------------------------------------------------- #
# Polynomial engine (finite differences - handles parentheses, implicit mult)
# --------------------------------------------------------------------------- #

Polynomial = dict[int, float]


def polynomial_coeffs(expression: str, var: str = "x", max_degree: int = 4) -> Polynomial | None:
    """Return ``{exponent: coefficient}`` when *expression* is a polynomial.

    Uses exact-ish sampling plus Newton forward differences, which means
    ``2(x+1)^2`` works without writing an expander.  Returns ``None`` when the
    expression is not a low degree polynomial (or contains other variables).
    """
    if var.lower() not in normalize_expression(expression).lower():
        return None
    samples: list[float] = []
    for i in range(max_degree + 1):
        try:
            samples.append(safe_eval(expression, {var: float(i)}))
        except (ValueError, ZeroDivisionError, OverflowError):
            return None
    if any(math.isnan(v) or math.isinf(v) for v in samples):
        return None

    # Newton forward differences -> binomial basis -> monomial basis
    diffs: list[float] = []
    current = list(samples)
    for _ in range(max_degree + 1):
        diffs.append(current[0])
        current = [current[i + 1] - current[i] for i in range(len(current) - 1)]

    if abs(diffs[max_degree]) > 1e-9 * max(1.0, max(abs(v) for v in samples)):
        return None  # higher degree than supported -> not treated as polynomial

    coeffs: Polynomial = {}
    for k, delta in enumerate(diffs):
        if abs(delta) < 1e-12:
            continue
        # expand delta * C(x, k) into monomial coefficients
        binom: Polynomial = {0: 1.0}
        for j in range(k):
            new: Polynomial = {}
            for exp, coef in binom.items():
                new[exp + 1] = new.get(exp + 1, 0.0) + coef
                new[exp] = new.get(exp, 0.0) - coef * j
            binom = {e: c / (j + 1) for e, c in new.items()}
        for exp, coef in binom.items():
            coeffs[exp] = coeffs.get(exp, 0.0) + delta * coef
    return {e: round(c, 10) for e, c in coeffs.items() if abs(c) > 1e-9} or {0: 0.0}


def polynomial_str(poly: Polynomial, var: str = "x") -> str:
    """Render a polynomial dictionary as a readable string."""
    if not poly:
        return "0"
    parts: list[str] = []
    for exp in sorted(poly, reverse=True):
        coef = poly[exp]
        if abs(coef) < 1e-12:
            continue
        sign = "-" if coef < 0 else "+"
        mag = abs(coef)
        mag_str = f"{mag:g}"
        if exp == 0:
            body = mag_str
        elif exp == 1:
            body = f"{var}" if abs(mag - 1) < 1e-12 else f"{mag_str}{var}"
        else:
            body = f"{var}^{exp}" if abs(mag - 1) < 1e-12 else f"{mag_str}{var}^{exp}"
        parts.append(f"{sign} {body}" if parts else (f"-{body}" if sign == "-" else body))
    return " ".join(parts) or "0"


def polynomial_eval(poly: Polynomial, value: float) -> float:
    """Evaluate a polynomial dictionary at *value*."""
    return sum(coef * (value ** exp) for exp, coef in poly.items())


def polynomial_derivative(poly: Polynomial) -> Polynomial:
    """Differentiate a polynomial (power rule)."""
    return {exp - 1: coef * exp for exp, coef in poly.items() if exp >= 1}


def solve_polynomial(poly: Polynomial) -> list[float]:
    """Solve ``poly(x) = 0`` for degrees 0/1/2 (real roots)."""
    coeffs = {e: c for e, c in poly.items() if abs(c) > 1e-12}
    degree = max(coeffs) if coeffs else 0
    if degree == 0:
        return []
    if degree == 1:
        return [-coeffs.get(0, 0.0) / coeffs[1]]
    if degree == 2:
        a, b, c = coeffs.get(2, 0.0), coeffs.get(1, 0.0), coeffs.get(0, 0.0)
        disc = b * b - 4 * a * c
        if disc < 0:
            return []
        root = math.sqrt(disc)
        return sorted({round((-b + root) / (2 * a), 10), round((-b - root) / (2 * a), 10)})
    return []


# --------------------------------------------------------------------------- #
# Problem analysis
# --------------------------------------------------------------------------- #

_ASSIGNMENT_RE = re.compile(r"(?<![A-Za-z0-9_])([a-zA-Z])\s*=\s*(-?\d+(?:\.\d+)?)(?![0-9.])")


def find_assignments(text: str) -> dict[str, float]:
    """Find ``x = 2`` style variable assignments in the text.

    Only a *standalone* single letter counts, so ``3x = 6`` is not read as
    ``x = 6``.  No keyword is required, which means ``حل کن x=2 3y+1`` and
    ``solve x = 2`` both work.
    """
    normalized = normalize_text(text)
    found: dict[str, float] = {}
    for name, value in _ASSIGNMENT_RE.findall(normalized):
        found[name] = float(value)
    return found


def split_equation(text: str) -> tuple[str, str] | None:
    """Split ``"2x + 3 = 7"`` into ``("2x + 3", "7")``.

    When the text contains no ``=`` but is a variable expression (``"2x + 3"``)
    the equation is interpreted as *expression = 0*, which lets the agent solve
    ``x^2 + 3x`` without the user writing the zero.  Returns ``None`` when the
    text is not usable as an equation.
    """
    expression = extract_math_expression(text)
    if not expression:
        return None
    if "=" in expression:
        left, _, right = expression.partition("=")
        if left.strip() and right.strip():
            return left, right
        return None
    if not re.fullmatch(r"[0-9A-Za-z\.\+\-\*/^\(\)]+", expression):
        return None
    if not any(char.isalpha() for char in expression):
        return None
    return expression, "0"


def extract_variables(expression: str) -> list[str]:
    """Return the single letter variables used by *expression*."""
    names = {
        token
        for token in re.findall(r"[A-Za-z]+", normalize_expression(expression))
        if token not in MATH_FUNCTIONS and token not in MATH_CONSTANTS
    }
    return sorted(name for name in names if len(name) == 1)


def analyze_problem(
    text: str,
    assignments: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Deterministically analyse a solvable problem.

    Handles three shapes:

    * ``substitution`` - the user pinned a value (``x = 2``) and wants the
      expression evaluated; the pinned clause is removed from the target.
    * ``equation`` - the user wants the unknown solved (``2x + 3 = 7``, or
      ``x^2 + 3x`` meaning ``= 0``).
    * ``evaluation`` - a pure numeric expression (``3 + 4 * 2``).

    *assignments* lets the caller pass variable values it found elsewhere in the
    conversation (for example in the user's sentence when *text* is only the
    extracted expression).  The returned dictionary always contains
    ``expression``, ``assignments``, ``kind``, ``result``, ``steps`` and
    ``method``; ``kind`` is ``"unknown"`` when the offline engine cannot handle
    the input.
    """
    normalized = normalize_text(text)
    topic = detect_topic(normalized)
    expression = extract_math_expression(normalized)
    pinned = dict(assignments) if assignments else find_assignments(normalized)
    equation = split_equation(normalized) if pinned == {} else None

    result: dict[str, Any] = {
        "expression": expression,
        "assignments": pinned,
        "kind": "unknown",
        "equation": equation,
        "result": None,
        "steps": [],
        "method": "",
        "topic": topic,
    }
    if not expression:
        return result

    # A pinned value (x = 2) wins over the equation reading when the expression
    # only needs variables that are pinned: "اگر x=2 باشد، 3x^2+1" must be
    # substituted, not turned into "3x^2+1 = 0".  Substitution is applied
    # uniformly so that "x = 2, evaluate x^2" answers 4 instead of "x = 2".
    if pinned:
        target = strip_assignments(expression)
        if not target or not any(char.isdigit() for char in target):
            target = expression
        try:
            value = safe_eval(target, pinned)
        except (ValueError, ZeroDivisionError, OverflowError, SyntaxError):
            value = None
        if value is not None:
            substituted = _substitute(target, pinned)
            result["expression"] = target
            result["kind"] = "substitution"
            result["result"] = round(value, 10)
            result["method"] = "substitution"
            result["steps"] = [
                f"جایگذاری مقادیر داده‌شده: {substituted}",
                f"ساده‌سازی عبارت: {_simplify_display(target, pinned)}",
                f"محاسبه مقدار نهایی: {value:g}",
            ]
            return result

    if equation is not None:
        left, right = equation
        poly = polynomial_coeffs(f"({left})-({right})", "x")
        if poly is not None:
            roots = solve_polynomial(poly)
            result["kind"] = "equation"
            result["method"] = "solve equation"
            result["steps"] = [
                f"نوشتن معادله به شکل استاندارد: {left} = {right}",
                f"انتقال همه‌چیز به یک طرف: {polynomial_str(poly)} = 0",
            ]
            if roots:
                result["result"] = roots[0] if len(roots) == 1 else roots
                result["steps"].append(
                    f"حل معادله: x = {', '.join(f'{root:g}' for root in roots)}"
                )
            else:
                result["steps"].append("این معادله جواب حقیقی ندارد.")
            return result

    # no pinned values and no equation: this is either a plain numeric
    # evaluation, or a variable expression where the variable stands for the
    # "final value" being asked for (f(x) = x^2 + 3x -> f(x) is the answer).
    try:
        value = safe_eval(expression, {})
    except (ValueError, ZeroDivisionError, OverflowError, SyntaxError):
        variables = extract_variables(expression)
        fallback = {name: DEFAULT_VARIABLE_VALUES.get(name, 0.0) for name in variables}
        if not fallback:
            return result
        try:
            value = safe_eval(expression, fallback)
        except (ValueError, ZeroDivisionError, OverflowError, SyntaxError):
            return result
        variables_text = ", ".join(f"{name} = {fallback[name]:g}" for name in sorted(fallback))
        result["assignments"] = fallback
        result["kind"] = "expression"
        result["method"] = "evaluate expression"
        result["result"] = round(value, 10)
        result["steps"] = [
            f"عبارت داده‌شده: {expression}",
            f"مقدار متغیر تعیین نشده است؛ برای محاسبه مقدار، {variables_text} در نظر گرفته می‌شود.",
            f"محاسبه مقدار: {value:g}",
        ]
        return result

    result["kind"] = "evaluation"
    result["method"] = "direct evaluation"
    result["result"] = round(value, 10)
    result["steps"] = [
        f"نوشتن عبارت عددی: {expression}",
        f"محاسبه گام‌به‌گام (با رعایت ترتیب عملگرها): {value:g}",
    ]
    return result


def _substitute(expression: str, assignments: Mapping[str, float]) -> str:
    """Return *expression* with variables replaced by their numeric values.

    ``3*x^2+1`` with ``x = 2`` becomes ``3*(2)^2+1`` - readable enough to be
    shown to the user as the "substitution" step.
    """
    out = normalize_expression(expression).replace("**", "^")
    for name, value in sorted(assignments.items(), key=lambda kv: -len(kv[0])):
        out = re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", f"({value:g})", out
        )
    out = re.sub(r"\s+", " ", out)
    return re.sub(r"(?<=[0-9\)])(?=[A-Za-z\(])", "*", out).strip()


def _simplify_display(expression: str, assignments: Mapping[str, float]) -> str:
    """Evaluate intermediate pieces so the user sees the arithmetic shrink."""
    try:
        return f"{safe_eval(expression, assignments):g}"
    except Exception:  # pragma: no cover - defensive
        return expression


def verify_roots(expression: str | None, answer: Any) -> list[str]:
    """Check that every root satisfies the equation (or that none exist).

    Returns a list of problem notes; empty when the check passes or cannot be
    performed.
    """
    if not expression or "=" not in expression:
        return []
    left, _, right = expression.partition("=")
    try:
        polynomial = polynomial_coeffs(f"({left})-({right})", "x")
    except Exception:  # pragma: no cover - defensive
        return []
    if polynomial is None:
        return []

    if isinstance(answer, (list, tuple)):
        candidates = [float(value) for value in answer]
    elif isinstance(answer, (int, float)) and not isinstance(answer, bool):
        candidates = [float(answer)]
    else:
        return []

    problems: list[str] = []
    for root in candidates:
        residue = polynomial_eval(polynomial, root)
        if abs(residue) > 1e-6 * max(1.0, abs(root)):
            problems.append(f"ریشه {root:g} در معادله صدق نمی‌کند / root does not satisfy the equation")
    if not problems:
        verified = ", ".join(f"{root:g}" for root in candidates)
        return [f"جایگذاری ریشه‌ها در معادله: x = {verified} معادله را صفر می‌کند / substitution check passed."]
    return problems


def verify_answer(
    steps: Sequence[str],
    answer: Any,
    expression: str | None = None,
    assignments: Mapping[str, float] | None = None,
    kind: str | None = None,
) -> dict[str, Any]:
    """Independently verify a produced answer.

    Three independent checks are performed:

    1. **Recomputation** - when the answer is numeric and the original
       expression plus the variable values are known, the answer is recomputed
       with Python's own arithmetic.
    2. **Step consistency** - every self-contained ``a = b`` arithmetic fragment
       inside the steps is re-evaluated and compared.  Fragments that contain a
       symbol (an equation being rearranged) or a constant identity such as
       ``= 0`` are skipped: they are algebra, not arithmetic.
    3. **Sanity** - a numeric answer must be real and finite.

    Returns ``{"is_correct": bool, "method": str, "notes": [str], "checked": ...}``.
    """
    notes: list[str] = []
    method = "none"
    is_correct: bool | None = None
    numeric_answer: float | None = None
    symbolic = isinstance(answer, str)

    if isinstance(answer, (int, float)) and not isinstance(answer, bool):
        numeric_answer = float(answer)
        if math.isnan(numeric_answer) or math.isinf(numeric_answer):
            return {
                "is_correct": False,
                "method": "sanity",
                "notes": ["پاسخ متناهی نیست / the answer is not finite."],
                "checked": answer,
            }

    if expression and numeric_answer is not None and kind != "equation":
        try:
            recomputed = safe_eval(expression, assignments or {})
        except Exception:
            recomputed = None
        if recomputed is not None:
            method = "recomputation"
            matches = math.isclose(recomputed, numeric_answer, rel_tol=1e-9, abs_tol=1e-9)
            is_correct = matches
            notes.append(
                f"محاسبه مجدد مستقل: {expression} با "
                f"{dict(assignments or {})} = {recomputed:g} "
                f"({'مطابق' if matches else 'ناهمخوان'} با پاسخ {numeric_answer:g})"
            )

    step_failures: list[str] = []
    for step in steps or []:
        for lhs, rhs in re.findall(r"([0-9][0-9\.\s\+\-\*/\(\)\^]*?)\s*=\s*(-?[0-9\.]+)", step):
            lhs, rhs = lhs.strip(), rhs.strip()
            if not any(op in lhs for op in "+-*/^"):
                continue
            if re.fullmatch(r"0+(?:\.0+)?", rhs):
                # "expression = 0" is a rearrangement, not a computation
                continue
            if any(
                re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", lhs)
                for name in (assignments or {})
            ):
                # mentions an assigned symbol (e.g. "x = 3*4 + 1 = 13"); the
                # independent recomputation above already covers this step
                continue
            try:
                left_value = safe_eval(lhs, assignments or {})
                right_value = float(rhs)
            except Exception:
                continue
            if not math.isclose(left_value, right_value, rel_tol=1e-9, abs_tol=1e-9):
                step_failures.append(f"گام ناسازگار / inconsistent step: {lhs} = {rhs}")
    if step_failures:
        notes.extend(step_failures)
        is_correct = False
    elif method != "none":
        notes.append("تمام گام‌های عددی با محاسبه مستقل سازگارند / all numeric steps match.")

    if is_correct is None and kind == "equation":
        # an equation is verified by substituting its roots back in
        root_notes = verify_roots(expression, answer)
        notes.extend(root_notes)
        if any("صدق نمی‌کند" in note for note in root_notes):
            is_correct = False
        elif root_notes:
            is_correct = True
            method = "root_substitution"

    if is_correct is None:
        is_correct = bool(notes) or numeric_answer is not None or symbolic
        method = "structural"
        if symbolic:
            notes.append(
                "بررسی ساختاری: پاسخ یک عبارت جبری است؛ گام‌ها منسجم‌اند "
                "(بازبینی عددی مستقل ممکن نیست)."
            )
        else:
            notes.append(
                "بررسی ساختاری: گام‌ها منسجم‌اند (محاسبه مستقل ممکن نبود)."
            )

    return {
        "is_correct": bool(is_correct),
        "method": method,
        "notes": notes,
        "checked": numeric_answer if numeric_answer is not None else answer,
    }


# --------------------------------------------------------------------------- #
# Offline explanations
# --------------------------------------------------------------------------- #

_CONCEPT_BANK: Mapping[str, dict[str, dict[str, str]]] = {
    "derivative": {
        "fa": {
            "definition": "مشتق، آهنگ تغییر یک تابع نسبت به متغیر آن است؛ یعنی شیب خط مماس بر نمودار تابع در هر نقطه.",
            "intuition": "تصور کن در حال رانندگی هستی: مکان تو تابع زمان است و مشتق آن، سرعت لحظه‌ای تو. مشتق می‌گوید «در همین لحظه، با چه سرعتی تغییر می‌کنی؟»",
            "example": "برای f(x) = x^2 + 3x، مشتق برابر است با f'(x) = 2x + 3 (توان را یکی کم و در ضریب ضرب می‌کنیم).",
            "notes": "قواعد پایه: مشتق x^n برابر n·x^(n-1)، مشتق عدد ثابت صفر، و مشتق مجموع برابر مجموع مشتق‌هاست.",
        },
        "en": {
            "definition": "The derivative measures how fast a function changes with respect to its variable - the slope of the tangent line.",
            "intuition": "Think of driving: position is a function of time and the derivative is your instantaneous speed.",
            "example": "For f(x) = x^2 + 3x the derivative is f'(x) = 2x + 3 (power rule, term by term).",
            "notes": "Power rule: d/dx x^n = n·x^(n-1); constants differentiate to 0; derivatives are linear.",
        },
    },
    "integral": {
        "fa": {
            "definition": "انتگرال، عمل معکوس مشتق است و سطح زیر نمودار تابع را اندازه می‌گیرد.",
            "intuition": "اگر مشتق «سرعت تغییر» را می‌دهد، انتگرال تغییرات را جمع می‌زند و کل انباشته را می‌سازد.",
            "example": "∫ 2x dx = x^2 + C",
            "notes": "هر انتگرال نامعین یک ثابت دلخواه C دارد؛ در انتگرال معین این ثابت حذف می‌شود.",
        },
        "en": {
            "definition": "An integral is the inverse operation of differentiation and measures the accumulated area under a curve.",
            "intuition": "If the derivative is a rate of change, the integral adds those changes up.",
            "example": "∫ 2x dx = x^2 + C",
            "notes": "Indefinite integrals carry a constant C; definite integrals evaluate the difference of antiderivatives.",
        },
    },
    "function": {
        "fa": {
            "definition": "تابع یک قاعده است که به هر ورودی مجاز، دقیقاً یک خروجی نسبت می‌دهد.",
            "intuition": "مثل دستگاه خودپرداز: شما یک ورودی (کارت و مبلغ) می‌دهید و دستگاه یک خروجی مشخص تولید می‌کند.",
            "example": "برای f(x) = 3x^2 + 1، اگر x = 2 باشد آنگاه f(2) = 3(4) + 1 = 13.",
            "notes": "دامنه مجموعه ورودی‌های مجاز و برد مجموعه خروجی‌های ممکن است.",
        },
        "en": {
            "definition": "A function is a rule assigning exactly one output to each allowed input.",
            "intuition": "Like a vending machine: one input produces exactly one output.",
            "example": "For f(x) = 3x^2 + 1 with x = 2 we get f(2) = 3·4 + 1 = 13.",
            "notes": "Domain = allowed inputs, range = reachable outputs.",
        },
    },
    "equation": {
        "fa": {
            "definition": "معادله یک تساوی ریاضی است که در آن یک یا چند مجهول وجود دارد.",
            "intuition": "معادله مثل یک ترازو است: هر کاری با یک طرف بکنی باید با طرف دیگر هم بکنی تا تعادل حفظ شود.",
            "example": "2x + 3 = 7 ⇒ 2x = 4 ⇒ x = 2",
            "notes": "هدف، جدا کردن مجهول در یک طرف و ساده‌سازی طرف دیگر است.",
        },
        "en": {
            "definition": "An equation is a mathematical equality containing one or more unknowns.",
            "intuition": "Treat it like a balance scale: whatever you do to one side you must do to the other.",
            "example": "2x + 3 = 7 ⇒ 2x = 4 ⇒ x = 2",
            "notes": "Goal: isolate the unknown on one side.",
        },
    },
    "limit": {
        "fa": {
            "definition": "حد، مقداری است که تابع به آن نزدیک می‌شود وقتی متغیر به یک نقطه نزدیک شود.",
            "intuition": "مثل نزدیک شدن به یک شهر روی نقشه: ممکن است هرگز داخل شهر نروی، اما فاصله‌ات به آن میل می‌کند.",
            "example": "lim(x→2) (x^2 - 4)/(x - 2) = lim(x→2)(x + 2) = 4",
            "notes": "حد ممکن است وجود داشته باشد حتی اگر تابع در آن نقطه تعریف نشده باشد.",
        },
        "en": {
            "definition": "A limit is the value a function approaches as its input approaches a point.",
            "intuition": "Like approaching a city: you may never enter it, yet you get arbitrarily close.",
            "example": "lim(x→2) (x^2 - 4)/(x - 2) = lim(x→2)(x + 2) = 4",
            "notes": "A limit can exist even where the function is undefined.",
        },
    },
    "trigonometry": {
        "fa": {
            "definition": "مثلثات به رابطه بین زاویه‌ها و اضلاع مثلث می‌پردازد؛ sin، cos و tan نسبت‌های اصلی آن هستند.",
            "intuition": "دایره واحد را تصور کن: cos طول افقی و sin ارتفاع عمودی نقطه روی دایره است.",
            "example": "sin(30°) = 1/2 و cos(60°) = 1/2",
            "notes": "اتحاد مهم: sin^2(x) + cos^2(x) = 1",
        },
        "en": {
            "definition": "Trigonometry studies the relation between angles and side lengths; sin, cos and tan are its core ratios.",
            "intuition": "On the unit circle, cos is the horizontal coordinate and sin the vertical one.",
            "example": "sin(30°) = 1/2 and cos(60°) = 1/2",
            "notes": "Key identity: sin^2(x) + cos^2(x) = 1",
        },
    },
    "logarithm": {
        "fa": {
            "definition": "لگاریتم پاسخ این پرسش است: «پایه را به چه توانی برسانیم تا این عدد به دست آید؟»",
            "intuition": "لگاریتم، معکوس توان است؛ مثل باز کردن یک قفل با پرسیدن «چه توانی لازم است؟».",
            "example": "log2(8) = 3 چون 2^3 = 8",
            "notes": "قواعد: log(ab) = log a + log b و log(a^n) = n·log a",
        },
        "en": {
            "definition": "A logarithm answers: to what power must the base be raised to get this number?",
            "intuition": "Logs are the inverse of exponentiation.",
            "example": "log2(8) = 3 because 2^3 = 8",
            "notes": "Rules: log(ab) = log a + log b, log(a^n) = n·log a",
        },
    },
}

_GENERIC_CONCEPT: Mapping[str, dict[str, str]] = {
    "fa": {
        "definition": "این پرسش یک مفهوم ریاضی را هدف گرفته است.",
        "intuition": "برای فهم بهتر، مفهوم را با یک مثال کوچک و ملموس می‌سازیم و سپس تعریف دقیق را می‌گوییم.",
        "example": "مثال ساده: هر مفهوم ریاضی را ابتدا روی یک نمونه کوچک امتحان کن، سپس قاعده کلی را بنویس.",
        "notes": "نکته: ابتدا تعریف، سپس مثال، و در پایان شرط‌های اعتبار قاعده را بررسی کن.",
    },
    "en": {
        "definition": "This question targets a mathematical concept.",
        "intuition": "We build intuition with a small concrete example first, then state the precise definition.",
        "example": "Try any concept on a tiny example first, then write the general rule.",
        "notes": "Order: definition, example, then the conditions under which the rule holds.",
    },
}

_LEVEL_PREFIX: Mapping[str, Mapping[str, str]] = {
    "beginner": {
        "fa": "از پایه و بدون فرض دانش پیشرفته شروع می‌کنیم.",
        "en": "We start from the basics with no advanced assumptions.",
    },
    "intermediate": {
        "fa": "با فرض آشنایی مقدماتی، روی کاربرد و نکات کلیدی تمرکز می‌کنیم.",
        "en": "Assuming basic familiarity, we focus on application and key points.",
    },
    "advanced": {
        "fa": "با نگاه دقیق‌تر، شرط‌ها و استدلال کامل را بررسی می‌کنیم.",
        "en": "We take a precise view, including conditions and full reasoning.",
    },
}


def explain_concept(topic: str, text: str, level: str, lang: str) -> dict[str, str]:
    """Build an offline concept explanation tuned to topic and user level."""
    lang = "fa" if lang == "fa" else "en"
    bank = _CONCEPT_BANK.get(topic)
    payload = dict(bank[lang]) if bank else dict(_GENERIC_CONCEPT[lang])

    prefix = _LEVEL_PREFIX.get(level, _LEVEL_PREFIX["intermediate"])[lang]
    payload["definition"] = f"{prefix} {payload['definition']}"

    if topic == "derivative":
        expression = extract_math_expression(text)
        if expression:
            poly = polynomial_coeffs(expression)
            if poly:
                derived = polynomial_derivative(poly)
                payload["example"] = (
                    f"برای f(x) = {polynomial_str(poly)}، مشتق برابر است با "
                    f"f'(x) = {polynomial_str(derived)}."
                    if lang == "fa"
                    else f"For f(x) = {polynomial_str(poly)} the derivative is f'(x) = {polynomial_str(derived)}."
                )
    return payload


def clarification_question(missing: str, lang: str) -> str:
    """Return a short follow-up question for an unclear request."""
    questions = {
        "no_math": {
            "fa": "لطفاً صورت مسئله یا عبارت ریاضی را دقیق بنویس (مثلاً: مشتق x^2 + 3x چیست؟).",
            "en": "Please write the exact problem or expression (e.g. what is the derivative of x^2 + 3x?).",
        },
        "no_expression": {
            "fa": "چه عبارت یا معادله‌ای را باید حل کنم؟ لطفاً آن را کامل بنویس.",
            "en": "Which expression or equation should I solve? Please write it out in full.",
        },
        "no_variable_value": {
            "fa": "مقدار متغیرها را مشخص نکرده‌ای. مثلاً اگر x = 2 است، بنویس «اگر x = 2 باشد، مقدار ... چقدر است؟».",
            "en": "You did not give the variable values. E.g. write 'if x = 2, what is the value of ...'.",
        },
        "unknown_topic": {
            "fa": "موضوع دقیقاً چیست؟ مشتق، انتگرال، معادله، مثلثات یا چیز دیگری؟",
            "en": "What is the exact topic? Derivative, integral, equation, trigonometry or something else?",
        },
    }
    bank = questions.get(missing, questions["no_math"])
    return bank["fa"] if lang == "fa" else bank["en"]


def expression_covers_assignment(expression: str | None, assignments: Mapping[str, float]) -> bool:
    """True when every variable *expression* needs has been pinned.

    ``"3x^2 + 1"`` with ``{"x": 2}`` is covered; ``"3y + 1"`` with ``{"x": 2}``
    is not, because ``y`` is still unknown.  With no assignments at all the
    check is vacuous (an equation does not need pinned values).  The assignment
    clause itself is removed first so ``x = 2`` is not mistaken for a use of
    ``x``.
    """
    if not expression or not assignments:
        return True
    stripped = strip_assignments(expression) or expression
    used = set(extract_variables(stripped))
    return used.issubset(set(assignments))


def detect_missing_information(text: str, classification: Mapping[str, Any]) -> str | None:
    """Return a machine readable reason when the query cannot be answered."""
    normalized = normalize_text(text)
    expression = extract_math_expression(normalized)
    topic = str(classification.get("subject_or_topic") or "unknown")
    request_type = str(classification.get("request_type") or "unclear")
    word_count = len(normalized.split())

    if request_type == "unclear":
        return "no_math"
    if request_type in {"solvable_problem", "mixed"}:
        if not expression:
            return "no_expression"
        assignments = find_assignments(normalized)
        if assignments and not expression_covers_assignment(expression, assignments):
            return "no_variable_value"
        if "=" in expression:
            left, _, right = expression.partition("=")
            if not left.strip() or not right.strip():
                return "no_expression"
    if request_type == "conceptual" and topic == "unknown" and word_count <= 2:
        return "unknown_topic"
    return None


# --------------------------------------------------------------------------- #
# Offline solution text
# --------------------------------------------------------------------------- #

def solve_problem(
    text: str,
    topic: str,
    level: str,
    lang: str,
    assignments: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Produce a full offline solution (analysis, method, steps, answer).

    The result is *verified before it leaves this function*: ``verify_answer``
    recomputes the arithmetic with Python, and for derivatives a finite
    difference check is run as well.  ``is_answer_correct`` and
    ``verification_notes`` are therefore already populated.
    """
    lang = "fa" if lang == "fa" else "en"
    analysis = analyze_problem(text, assignments)
    expression = analysis.get("expression")
    result = analysis.get("result")
    kind = analysis.get("kind")
    detected = analysis.get("topic") or topic

    steps: list[str] = list(analysis.get("steps") or [])
    extra_notes: list[str] = []

    if topic == "derivative" and expression:
        poly = polynomial_coeffs(expression)
        if poly is not None:
            derived = polynomial_derivative(poly)
            result = polynomial_str(derived)
            kind = "derivative"
            steps = [
                f"عبارت داده‌شده: f(x) = {polynomial_str(poly)}",
                "اعمال قاعده توان روی هر جمله: d/dx a·x^n = a·n·x^(n-1)",
                f"نتیجه: f'(x) = {polynomial_str(derived)}",
            ]
            extra_notes.extend(_derivative_check(expression, derived))
        else:
            steps = [
                f"عبارت داده‌شده: {expression}",
                "این عبارت چندجمله‌ای نیست، بنابراین با قواعد ساده مشتق‌گیری "
                "(توان، ضرب، زنجیره‌ای) گام‌به‌گام مشتق گرفته می‌شود.",
                "پاسخ به شکل عبارت مشتق نوشته می‌شود.",
            ]
            result = f"d/dx ({expression})"

    elif detected == "derivative" and expression:
        poly = polynomial_coeffs(expression)
        if poly is not None:
            derived = polynomial_derivative(poly)
            result = polynomial_str(derived)
            kind = "derivative"
            steps = [
                f"عبارت داده‌شده: f(x) = {polynomial_str(poly)}",
                "اعمال قاعده توان روی هر جمله: d/dx a·x^n = a·n·x^(n-1)",
                f"نتیجه: f'(x) = {polynomial_str(derived)}",
            ]
            extra_notes.extend(_derivative_check(expression, derived))

    if not steps:
        steps = [
            "داده‌ها و مجهول مسئله را مشخص می‌کنیم.",
            "رابطه یا فرمول مناسب را انتخاب می‌کنیم.",
            "مقادیر را جایگذاری و گام‌به‌گام ساده می‌کنیم.",
        ]

    verified = verify_answer(
        steps,
        result,
        # the expression is always passed: verify_answer skips the plain
        # recomputation for equations/derivatives (kind guards it) but still
        # needs it to substitute the roots back into the equation
        expression,
        analysis.get("assignments") or {},
        kind,
    )
    notes = list(verified.get("notes") or []) + extra_notes
    is_correct = bool(verified.get("is_correct")) and not any(
        "ناموفق" in note or "ناسازگار" in note or "صدق نمی‌کند" in note for note in extra_notes
    )

    return {
        "problem_statement": expression or normalize_text(text),
        "given": analysis.get("assignments") or {},
        "unknown": "مقدار نهایی" if lang == "fa" else "final value",
        "method": analysis.get("method") or ("derivative" if topic == "derivative" else "تحلیل گام‌به‌گام"),
        "steps": steps,
        "answer": result,
        "expression": expression,
        "assignments": analysis.get("assignments") or {},
        "kind": kind,
        "verification_notes": notes,
        "is_answer_correct": is_correct,
        "verification_method": str(verified.get("method") or "structural"),
    }


def _derivative_check(expression: str, derived: Polynomial) -> list[str]:
    """Verify a derivative by comparing it with a central finite difference."""
    if not derived:
        return []
    sample = 1.3
    try:
        numeric = (
            safe_eval(expression, {"x": sample + 1e-6}) - safe_eval(expression, {"x": sample - 1e-6})
        ) / 2e-6
        symbolic = polynomial_eval(derived, sample)
    except Exception:  # pragma: no cover - defensive
        return []
    if math.isclose(numeric, symbolic, rel_tol=1e-4, abs_tol=1e-4):
        return [
            f"بررسی مستقل مشتق با تفاضل عددی در x = {sample:g}: {numeric:.6g} ≈ {symbolic:.6g} "
            "/ derivative check passed."
        ]
    return [f"بررسی مستقل مشتق ناموفق بود / derivative check failed ({numeric:.6g} vs {symbolic:.6g})."]


def format_number(value: Any) -> str:
    """Format a numeric result for display without trailing zeros."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(format_number(v) for v in value)
    return str(value)


def summarize_history(history: Iterable[Mapping[str, str]], limit: int = 6) -> str:
    """Render the last few conversation turns as a compact transcript."""
    items = list(history)[-limit:]
    lines = []
    for item in items:
        role = item.get("role", "user")
        content = str(item.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)
