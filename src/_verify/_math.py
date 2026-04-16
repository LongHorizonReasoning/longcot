"""Verification for math templates."""

from __future__ import annotations

import json
import re
from tokenize import TokenError
from typing import Literal

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from .._parsing import extract_balanced_brackets, extract_last_balanced_brackets, extract_solution
from .._types import MathFallbackResult, MathVerifyOptions, Question
from ._fallback import call_gemini_json, gemini_api_key

CompareStatus = Literal["match", "mismatch", "unresolved", "textual"]

_TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

_FUNCTIONS = {
    "sqrt": sp.sqrt,
    "cbrt": sp.cbrt,
    "log": sp.log,
    "ln": sp.log,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "arctan": sp.atan,
    "abs": sp.Abs,
}
_CONSTANTS = {"pi": sp.pi, "e": sp.E}
_MATH_WORDS = set(_FUNCTIONS) | set(_CONSTANTS) | {"frac", "binom"}
_TEXTUAL_HINTS = (
    "all polynomials",
    "integers with",
    "starting with",
    "less than",
    "greater than",
    "hours",
    "minutes",
    "p.m.",
    "a.m.",
    "odd integers",
    "even integers",
    "if ",
    "otherwise",
    "such that",
    "for some",
)


def _balanced(text: str) -> bool:
    stack: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for ch in text:
        if ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
    return not stack


def _strip_outer_container(text: str) -> str:
    text = text.strip()
    if len(text) < 2:
        return text
    if (text[0], text[-1]) not in {("[", "]"), ("(", ")")}:
        return text
    inner = text[1:-1].strip()
    return inner if _balanced(inner) else text


def _split_top_level_csv(text: str) -> list[str]:
    text = _strip_outer_container(text)
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return parts


def _replace_latex_fracs(text: str) -> str:
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", text)
    return text


def _normalize_roots(text: str) -> str:
    atom = r"(\([^()]+\)|[A-Za-z0-9_.]+)"
    text = re.sub(rf"(\d+(?:\.\d+)?|\)|[A-Za-z])\s*√\s*{atom}", r"\1*sqrt(\2)", text)
    text = re.sub(rf"√\s*{atom}", r"sqrt(\1)", text)
    text = re.sub(rf"(\d+(?:\.\d+)?|\)|[A-Za-z])\s*∛\s*{atom}", r"\1*cbrt(\2)", text)
    text = re.sub(rf"∛\s*{atom}", r"cbrt(\1)", text)
    return text


def _normalize_component(component: str) -> str:
    s = component.strip()
    s = s.strip("'").strip('"')
    s = s.replace("$", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.replace("≤", "<=").replace("≥", ">=").replace("≠", "!=")
    s = s.replace("\\leq", "<=").replace("\\geq", ">=")
    s = s.replace("\\le", "<=").replace("\\ge", ">=")
    s = s.replace("\\pi", "pi")
    s = s.replace("π", "pi")
    s = s.replace("cuberoot", "cbrt")
    s = _replace_latex_fracs(s)
    s = _normalize_roots(s)
    s = re.sub(r"\^\s*\{([^{}]+)\}", r"^(\1)", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s.endswith("."):
        s = s[:-1].strip()
    return s


def _component_is_textual(component: str) -> bool:
    lower = component.lower()
    if any(token in lower for token in _TEXTUAL_HINTS):
        return True
    if ";" in component:
        return True
    if re.search(r"(?:<=|>=|<|>)", component):
        return True
    words = re.findall(r"[A-Za-z]+", lower)
    for word in words:
        if len(word) == 1:
            continue
        if word in _MATH_WORDS:
            continue
        return True
    return False


def _parse_expression(component: str) -> sp.Expr | None:
    expr_text = _normalize_component(component)
    if not expr_text:
        return None

    names = set(re.findall(r"[A-Za-z_]\w*", expr_text))
    symbol_names = sorted(n for n in names if n not in _MATH_WORDS)
    local_dict: dict[str, object] = dict(_FUNCTIONS)
    local_dict.update(_CONSTANTS)
    for name in symbol_names:
        local_dict[name] = sp.Symbol(name, real=True)

    try:
        return parse_expr(
            expr_text,
            local_dict=local_dict,
            transformations=_TRANSFORMATIONS,
            evaluate=True,
        )
    except (sp.SympifyError, SyntaxError, TypeError, ValueError, TokenError):
        return None


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _compare_component(expected: str, predicted: str) -> CompareStatus:
    expected_norm = _normalize_component(expected)
    predicted_norm = _normalize_component(predicted)

    if expected_norm == predicted_norm:
        return "match"

    expected_expr = _parse_expression(expected_norm)
    predicted_expr = _parse_expression(predicted_norm)

    if expected_expr is not None and predicted_expr is not None:
        try:
            if sp.simplify(expected_expr - predicted_expr) == 0:
                return "match"
        except TypeError:
            pass
        equals = expected_expr.equals(predicted_expr)
        if equals is True:
            return "match"
        if equals is False:
            return "mismatch"
        return "unresolved"

    if _compact(expected_norm) == _compact(predicted_norm):
        return "match"

    return "unresolved"


def _answer_components(value: object) -> list[str] | None:
    """Normalize a stored answer payload into math-answer components."""
    if isinstance(value, list):
        parts = [str(item).strip() for item in value]
        return parts if parts else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        return _split_top_level_csv(text)
    text = str(value).strip()
    return [text] if text else None


def _compare_components(expected_parts: list[str], predicted_parts: list[str]) -> CompareStatus:
    """Compare two answer-component lists entry-by-entry."""
    if len(expected_parts) != len(predicted_parts):
        return "mismatch"

    if any(_component_is_textual(part) for part in expected_parts + predicted_parts):
        return "textual"

    for expected_part, predicted_part in zip(expected_parts, predicted_parts):
        status = _compare_component(expected_part, predicted_part)
        if status != "match":
            return status
    return "match"


def _extract_predicted_math_components(response: str) -> list[str] | None:
    """Extract math answer components using list-only extraction flow.

    Order:
    1) If ``solution =`` exists, take the first balanced list after it.
    2) Otherwise, take the last balanced list in the whole response.
    """
    solution_text = extract_solution(response)
    if solution_text is not None:
        bracketed = extract_balanced_brackets(solution_text)
        if bracketed is None:
            return None
        return _split_top_level_csv(bracketed)

    fallback_list = extract_last_balanced_brackets(response)
    if fallback_list is None:
        return None
    return _split_top_level_csv(fallback_list)


def _build_fallback_prompt(question: Question, expected: str, predicted: str) -> str:
    payload = {
        "question_id": question.question_id,
        "question_prompt": question.prompt,
        "expected": expected,
        "predicted": predicted,
        "task": "Determine whether predicted is mathematically equivalent to expected. Small formatting differences or errors in predicted are acceptable.",
    }
    return (
        "You are a strict math answer equivalence judge.\n"
        "Return JSON with keys: equivalent (bool), reason (string).\n"
        "Use equivalent=true only when mathematically equivalent.\n\n"
        f"Input:\n{json.dumps(payload, ensure_ascii=False)}"
    )


def _parse_fallback_result(data: dict[str, object]) -> MathFallbackResult:
    equivalent = data.get("equivalent")
    reason = data.get("reason")

    if not isinstance(equivalent, bool):
        raise ValueError("fallback response missing boolean equivalent")
    if reason is None:
        reason = ""
    if not isinstance(reason, str):
        raise ValueError("fallback response reason must be a string")

    return MathFallbackResult(
        equivalent=equivalent,
        reason=reason.strip(),
    )


def _gemini_fallback_judge(
    *,
    question: Question,
    expected: str,
    predicted: str,
    model: str,
    timeout_s: float,
) -> MathFallbackResult:
    prompt = _build_fallback_prompt(question, expected, predicted)
    parsed = call_gemini_json(prompt=prompt, model=model, timeout_s=timeout_s)
    return _parse_fallback_result(parsed)


def _run_fallback(
    *,
    question: Question,
    expected: str,
    predicted: str,
    options: MathVerifyOptions,
) -> bool:
    if not options.enable_fallback:
        return False

    judge = options.fallback_judge or _gemini_fallback_judge
    if judge is _gemini_fallback_judge and gemini_api_key() is None:
        return False
    retries = max(0, options.fallback_max_retries)

    for attempt in range(retries + 1):
        try:
            result = judge(
                question=question,
                expected=expected,
                predicted=predicted,
                model=options.fallback_model,
                timeout_s=options.fallback_timeout_s,
            )
        except Exception:
            if attempt == retries:
                return False
            continue

        return result.equivalent

    return False


def verify_math(response: str, question: Question, options: MathVerifyOptions | None = None) -> bool:
    """Math verification with deterministic comparison and mandatory fallback on failure."""
    if question.answer is None:
        return False

    verify_options = options if options is not None else MathVerifyOptions()
    expected_parts = _answer_components(question.answer)
    if expected_parts is None:
        return False
    expected = ", ".join(expected_parts)

    predicted_parts = _extract_predicted_math_components(response)
    if predicted_parts is None:
        # If list extraction fails at all stages, let fallback adjudicate.
        return _run_fallback(
            question=question,
            expected=expected,
            predicted=response.strip(),
            options=verify_options,
        )

    status = _compare_components(expected_parts, predicted_parts)
    if status == "match":
        return True

    return _run_fallback(
        question=question,
        expected=expected,
        predicted=response.strip(),
        options=verify_options,
    )
