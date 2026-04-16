"""Verification for chemistry templates."""

from __future__ import annotations

import json

from rdkit import Chem

from .._parsing import (
    extract_last_balanced_brackets,
    parse_list_from_text,
    parse_mixed_list_from_text,
    extract_solution,
)
from .._types import ChemistryFallbackResult, ChemistryVerifyOptions, Question
from ._fallback import call_gemini_json, gemini_api_key
from ._regexes import SMILES_ANYWHERE_RE, SMILES_FROM_START_RE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_smiles(smiles: str) -> str | None:
    """Canonicalize a SMILES string using RDKit.

    Tries the raw string first, then strips colon notation
    (e.g. ``C:C:C`` -> ``CCC``) for hard3-style answers.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol)

    stripped = smiles.replace(":", "")
    mol = Chem.MolFromSmiles(stripped)
    if mol is not None:
        return Chem.MolToSmiles(mol)

    return None


def _smiles_equivalent(s1: str, s2: str) -> bool:
    """Compare two SMILES strings for chemical equivalence."""
    s1, s2 = s1.strip(), s2.strip()
    if s1 == s2:
        return True

    c1 = _normalize_smiles(s1)
    c2 = _normalize_smiles(s2)
    if c1 is not None and c2 is not None:
        return c1 == c2

    return False


def _grab_smiles_from_start(text: str, i: int) -> str | None:
    """Extract a SMILES-like token anchored at index ``i``."""
    match = SMILES_FROM_START_RE.match(text, i)
    return match.group("smiles") if match else None


def _grab_last_smiles_anywhere(text: str) -> str | None:
    """Extract the last SMILES-like token appearing in ``text``."""
    matches = list(SMILES_ANYWHERE_RE.finditer(text))
    if not matches:
        return None
    return matches[-1].group("smiles")


def _extract_smiles_candidate(response: str) -> str | None:
    """Extract a SMILES candidate from model output.

    1) If ``solution =`` exists, anchor at the first non-whitespace character after ``=``.
    2) Otherwise (or if anchored parse fails), take the last SMILES-like token in the response.
    """
    extracted = extract_solution(response)
    if extracted is not None:
        candidate = _grab_smiles_from_start(extracted, 0)
        if candidate:
            return candidate
    return _grab_last_smiles_anywhere(response)


def _extract_list_candidate_text(response: str) -> str | None:
    """Extract list-bearing text, tolerating missing ``solution =`` labels."""
    extracted = extract_solution(response)
    if extracted is not None:
        return extracted
    # If no explicit solution label exists, use the last list-like block.
    return extract_last_balanced_brackets(response)


def _build_smiles_fallback_prompt(question: Question, response: str) -> str:
    payload = {
        "question_id": question.question_id,
        "question_prompt": question.prompt,
        "response_text": response,
        "task": (
            "Extract the final answer SMILES string from the response text. "
            "If no valid final-answer SMILES is present, return null."
        ),
    }
    return (
        "You are a strict chemistry answer extractor.\n"
        "Return JSON with keys: smiles (string or null), reason (string).\n"
        "Do not include any extra keys.\n\n"
        f"Input:\n{json.dumps(payload, ensure_ascii=False)}"
    )


def _parse_fallback_result(data: dict[str, object]) -> ChemistryFallbackResult:
    smiles = data.get("smiles")
    reason = data.get("reason")

    if smiles is not None and not isinstance(smiles, str):
        raise ValueError("fallback response smiles must be a string or null")
    if reason is None:
        reason = ""
    if not isinstance(reason, str):
        raise ValueError("fallback response reason must be a string")

    cleaned = smiles.strip() if isinstance(smiles, str) else None
    if cleaned == "":
        cleaned = None

    return ChemistryFallbackResult(smiles=cleaned, reason=reason.strip())


def _gemini_fallback_extractor(
    *,
    question: Question,
    response: str,
    model: str,
    timeout_s: float,
) -> ChemistryFallbackResult:
    prompt = _build_smiles_fallback_prompt(question, response)
    parsed = call_gemini_json(prompt=prompt, model=model, timeout_s=timeout_s)
    return _parse_fallback_result(parsed)


def _run_smiles_fallback(
    *,
    response: str,
    question: Question,
    options: ChemistryVerifyOptions,
) -> str | None:
    if not options.enable_fallback:
        return None

    judge = options.fallback_judge or _gemini_fallback_extractor
    if judge is _gemini_fallback_extractor and gemini_api_key() is None:
        return None
    retries = max(0, options.fallback_max_retries)

    for attempt in range(retries + 1):
        try:
            result = judge(
                question=question,
                response=response,
                model=options.fallback_model,
                timeout_s=options.fallback_timeout_s,
            )
        except Exception:
            if attempt == retries:
                return None
            continue
        return result.smiles

    return None



def _lists_equal_int(a: list, b: list) -> bool:
    """Recursively compare two (possibly nested) lists of ints."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if isinstance(x, list) and isinstance(y, list):
            if not _lists_equal_int(x, y):
                return False
        elif x != y:
            return False
    return True


def _mixed_lists_equal(extracted: list, answer: list) -> bool:
    """Element-wise comparison for hard2-style ``[SMILES, float, ...]``."""
    if len(extracted) != len(answer):
        return False
    for ext, ans in zip(extracted, answer):
        if isinstance(ans, str):
            if not isinstance(ext, str):
                return False
            if not _smiles_equivalent(ext, ans):
                return False
        elif isinstance(ans, (int, float)):
            try:
                if round(float(ext), 2) != round(float(ans), 2):
                    return False
            except (TypeError, ValueError):
                return False
        else:
            if ext != ans:
                return False
    return True


def _parse_string_list(text: str) -> list[str] | None:
    """Parse a flat list of strings, tolerating quoted and bare tokens."""
    parsed = parse_mixed_list_from_text(text)
    if parsed is None:
        return None
    out: list[str] = []
    for item in parsed:
        if isinstance(item, list):
            return None
        if isinstance(item, (int, float)):
            return None
        value = str(item).strip()
        if not value:
            return None
        out.append(value)
    return out


def _lists_equal_str(a: list[str], b: list[str]) -> bool:
    """Compare two flat string lists with trimmed element equality."""
    if len(a) != len(b):
        return False
    return all(x.strip() == y.strip() for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Template verifiers
# ---------------------------------------------------------------------------

def verify_smiles(
    response: str,
    question: Question,
    options: ChemistryVerifyOptions | None = None,
) -> bool:
    """SMILES comparison via regex extraction + RDKit canonicalization."""
    if not isinstance(question.answer, str):
        return False

    verify_options = options if options is not None else ChemistryVerifyOptions()
    extracted = _extract_smiles_candidate(response)
    if extracted is not None and _smiles_equivalent(extracted, question.answer):
        return True

    # Extraction can be wrong; if fallback is enabled, give the LLM extractor a chance.
    fallback_smiles = _run_smiles_fallback(
        response=response,
        question=question,
        options=verify_options,
    )
    if fallback_smiles is None:
        return False

    return _smiles_equivalent(fallback_smiles, question.answer)


def verify_int_list(response: str, question: Question) -> bool:
    """Flat integer list comparison (med1, med2)."""
    extracted = _extract_list_candidate_text(response)
    if extracted is None:
        return False
    parsed = parse_list_from_text(extracted, coerce_string_ints=True)
    if parsed is None:
        return False
    return _lists_equal_int(parsed, question.answer)



def verify_mixed_list(response: str, question: Question) -> bool:
    """Mixed SMILES + float list comparison (hard2)."""
    extracted = _extract_list_candidate_text(response)
    if extracted is None:
        return False
    parsed = parse_mixed_list_from_text(extracted)
    if parsed is None:
        return False
    return _mixed_lists_equal(parsed, question.answer)


def verify_string_list(response: str, question: Question) -> bool:
    """Flat string-list comparison (e.g. ['ASN', 'GLU', ...])."""
    extracted = _extract_list_candidate_text(response)
    if extracted is None:
        return False
    parsed = _parse_string_list(extracted)
    if parsed is None:
        return False
    if not isinstance(question.answer, list):
        return False
    expected = [str(x).strip() for x in question.answer]
    return _lists_equal_str(parsed, expected)
